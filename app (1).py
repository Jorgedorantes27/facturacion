
import os, re
from io import BytesIO
from datetime import datetime
from unicodedata import normalize as uni_normalize
import pandas as pd, numpy as np, streamlit as st

st.set_page_config(page_title="Cruce de Facturas vs Suscriptores", layout="wide")

# ---------------------- Auth opcional ----------------------
def check_auth():
    secret_pwd = st.secrets.get("APP_PASSWORD", None)
    env_pwd = os.getenv("APP_PASSWORD")
    password = secret_pwd or env_pwd
    if not password:
        return True  # sin password configurada
    if st.session_state.get("authed", False):
        return True
    st.info("La app está protegida. Ingresa la contraseña para continuar.")
    pwd_in = st.text_input("Contraseña", type="password", key="pwd_in")
    if st.button("Entrar", use_container_width=True):
        if pwd_in == password:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")
    st.stop()

check_auth()

# ---------------------- Utilidades ----------------------
def sanitize(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+","", str(name).upper())

def find_col(df: pd.DataFrame, targets):
    if isinstance(targets,str): targets=[targets]
    targets_san=[sanitize(t) for t in targets]
    san_map={c:sanitize(c) for c in df.columns}
    # exact match
    for c,s in san_map.items():
        if s in targets_san: return c
    # contains
    for c,s in san_map.items():
        if any(t in s for t in targets_san): return c
    return None

def norm_text(x: str) -> str:
    if pd.isna(x): return ""
    y = uni_normalize("NFKD", str(x)).encode("ascii","ignore").decode("ascii")
    y = y.upper().strip()
    y = re.sub(r"[\s\-]","", y)
    return y

def prep_facturas(df: pd.DataFrame) -> pd.DataFrame:
    rfc_col = find_col(df, ["RFC"])
    total_col = find_col(df, ["TOTAL"])
    numero_col = find_col(df, ["NUMERO","NÚMERO"])
    if rfc_col: df = df.rename(columns={rfc_col:"RFC_FACTURA"})
    if total_col: df = df.rename(columns={total_col:"MONTO_FACTURA"})
    if numero_col: df = df.rename(columns={numero_col:"NUMERO"})
    df["RFC_FACTURA_N"] = df["RFC_FACTURA"].map(norm_text)
    df["MONTO_FACTURA"] = pd.to_numeric(df["MONTO_FACTURA"], errors="coerce")
    df["MONTO_FACTURA_R"] = df["MONTO_FACTURA"].round(2)
    if "Fecha" in df.columns: df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    return df

def prep_suscriptores(df: pd.DataFrame, sheet_name: str, oficina="CDMX", estatus="ACTIVO") -> pd.DataFrame:
    rfc_col = find_col(df, ["SUS_RFC","RFC_SUS"])
    costo_col = find_col(df, ["SUS_COSTO","MONTO_SUS","COSTO"])
    giro_col = find_col(df, ["SUS_GIRO","GIRO"])
    oficina_col = find_col(df, ["SUS_OFICINA","OFICINA"])
    estatus_col = find_col(df, ["ESTATUS","STATUS"])
    sus_id_col = find_col(df, ["SUS_ID","ID"])
    nombre_col = find_col(df, ["SUS_NOMBRE","NOMBRE"])
    razon_col = find_col(df, ["SUS_RAZON_SOCIAL","RAZON SOCIAL","RAZÓN SOCIAL"])

    missing=[n for n,c in {"SUS_RFC":rfc_col,"SUS_COSTO":costo_col,"SUS_GIRO":giro_col,"SUS_OFICINA":oficina_col,"ESTATUS":estatus_col}.items() if c is None]
    if missing: raise ValueError(f"Faltan columnas requeridas en '{sheet_name}': {', '.join(missing)}")

    ren={rfc_col:"RFC_SUS", costo_col:"MONTO_SUS", giro_col:"GIRO", oficina_col:"SUS_OFICINA", estatus_col:"ESTATUS"}
    if sus_id_col: ren[sus_id_col]="sus_id"
    if nombre_col: ren[nombre_col]="SUS_NOMBRE"
    if razon_col: ren[razon_col]="SUS_RAZON_SOCIAL"
    df=df.rename(columns=ren)

    # Filtros previos obligatorios
    df["_OFI_N"]=df["SUS_OFICINA"].astype(str).str.upper().str.strip()
    df["_EST_N"]=df["ESTATUS"].astype(str).str.upper().str.strip()
    df=df[(df["_OFI_N"]==oficina.upper()) & (df["_EST_N"]==estatus.upper())].copy()

    # Solo GIRO = SI
    df["GIRO_N"]=df["GIRO"].astype(str).str.upper().str.strip()
    df=df[df["GIRO_N"]=="SI"].copy()

    df["MONTO_SUS"]=pd.to_numeric(df["MONTO_SUS"], errors="coerce")
    df["RFC_SUS"]=df["RFC_SUS"].astype(str)
    df["RFC_SUS_N"]=df["RFC_SUS"].map(norm_text)
    return df

def asignar_matches(sus_df: pd.DataFrame, fact_df: pd.DataFrame, iva=0.16, tol=1.0) -> pd.DataFrame:
    cols_keep=["sus_id","SUS_NOMBRE","SUS_RAZON_SOCIAL","RFC_SUS","RFC_SUS_N","MONTO_SUS","GIRO","GIRO_N"]
    for c in cols_keep:
        if c not in sus_df.columns: sus_df[c]=np.nan
    res=sus_df[cols_keep].copy()
    res["ESTADO_PAGO"]="No pagado"; res["FACTURA_NUMERO"]=np.nan; res["FACTURA_FOLIO"]=np.nan; res["FACTURA_MONTO"]=np.nan; res["CRITERIO"]=""

    facturas_by_rfc={rfc:df for rfc,df in fact_df.groupby("RFC_FACTURA_N")}

    def asignar_por_monto_con_iva(sub_rows: pd.DataFrame, facturas_rfc: pd.DataFrame)->pd.DataFrame:
        out=sub_rows.copy()
        pool=facturas_rfc[["NUMERO","Folio","MONTO_FACTURA_R","MONTO_FACTURA"]].copy()
        if "NUMERO" not in pool.columns: pool["NUMERO"]=np.nan
        pool["USADA"]=False
        for idx,row in out.iterrows():
            m_net=round((row["MONTO_SUS"] or 0),2)
            m_iva=round(m_net*(1+iva),2)
            cand_net=pool[~pool["USADA"] & (pool["MONTO_FACTURA_R"].sub(m_net).abs()<=tol)].copy()
            cand_iva=pool[~pool["USADA"] & (pool["MONTO_FACTURA_R"].sub(m_iva).abs()<=tol)].copy()
            best=None; criterio=None
            if not cand_net.empty:
                cand_net["diff"]=cand_net["MONTO_FACTURA_R"].sub(m_net).abs()
                best_net=cand_net.sort_values(["diff","MONTO_FACTURA_R"]).iloc[0]
                best=best_net; criterio=f"match neto ±{tol:.2f}"
            if not cand_iva.empty:
                cand_iva["diff"]=cand_iva["MONTO_FACTURA_R"].sub(m_iva).abs()
                best_iva=cand_iva.sort_values(["diff","MONTO_FACTURA_R"]).iloc[0]
                if best is None or best_iva["diff"]<best["diff"]:
                    best=best_iva; criterio=f"match con IVA ({int(iva*100)}%) ±{tol:.2f}"
            if best is not None:
                out.at[idx,"ESTADO_PAGO"]="Pagado"
                out.at[idx,"FACTURA_NUMERO"]=best.get("NUMERO",np.nan)
                out.at[idx,"FACTURA_FOLIO"]=best.get("Folio",np.nan)
                out.at[idx,"FACTURA_MONTO"]=best.get("MONTO_FACTURA_R",np.nan)
                out.at[idx,"CRITERIO"]=criterio
                pool.loc[pool.index==best.name,"USADA"]=True
        return out

    res_list=[]
    for rfc, sub_df in res.groupby("RFC_SUS_N"):
        fact_rfc=facturas_by_rfc.get(rfc)
        if fact_rfc is None or fact_rfc.empty: res_list.append(sub_df); continue

        suma_net=round(sub_df["MONTO_SUS"].sum(),2); suma_iva=round(suma_net*(1+iva),2)
        cand_net=fact_rfc[(fact_rfc["MONTO_FACTURA_R"]-suma_net).abs()<=tol]
        cand_iva=fact_rfc[(fact_rfc["MONTO_FACTURA_R"]-suma_iva).abs()<=tol]

        if not cand_net.empty or not cand_iva.empty:
            best=cand_iva.iloc[0] if not cand_iva.empty else cand_net.iloc[0]
            criterio=f"factura unica = suma cuentas {'con IVA' if not cand_iva.empty else 'neta'} ±{tol:.2f}"
            tmp=sub_df.copy(); tmp["ESTADO_PAGO"]="Pagado"
            tmp["FACTURA_NUMERO"]=best.get("NUMERO",np.nan); tmp["FACTURA_FOLIO"]=best.get("Folio",np.nan); tmp["FACTURA_MONTO"]=best.get("MONTO_FACTURA_R",np.nan)
            tmp["CRITERIO"]=criterio; res_list.append(tmp); continue

        res_list.append(asignar_por_monto_con_iva(sub_df, fact_rfc))

    return pd.concat(res_list, axis=0).reset_index(drop=True) if res_list else pd.DataFrame(columns=cols_keep+["ESTADO_PAGO","FACTURA_NUMERO","FACTURA_FOLIO","FACTURA_MONTO","CRITERIO"])

def build_excel(detalle: pd.DataFrame, facturas_df: pd.DataFrame, iva: float) -> bytes:
    # Traer fechas por NUMERO y fallback por Folio
    fdf=facturas_df.copy()
    if "NUMERO" not in fdf.columns and "Número" in fdf.columns: fdf=fdf.rename(columns={"Número":"NUMERO"})
    if "NUMERO" not in fdf.columns and "Numero" in fdf.columns: fdf=fdf.rename(columns={"Numero":"NUMERO"})
    det=detalle.merge(fdf[["NUMERO","Fecha"]], left_on="FACTURA_NUMERO", right_on="NUMERO", how="left").rename(columns={"Fecha":"FACTURA_FECHA"}).drop(columns=["NUMERO"], errors="ignore")
    if "Folio" in fdf.columns:
        falt=det["FACTURA_FECHA"].isna()
        folio_map=fdf[["Folio","Fecha"]].copy().rename(columns={"Fecha":"FACTURA_FECHA_FOLIO"})
        det=det.merge(folio_map, left_on="FACTURA_FOLIO", right_on="Folio", how="left")
        det.loc[falt,"FACTURA_FECHA"]=det.loc[falt,"FACTURA_FECHA_FOLIO"]
        det=det.drop(columns=["Folio","FACTURA_FECHA_FOLIO"], errors="ignore")

    # 1) NO PAGADAS (sin fecha)
    no_pag=det[det["ESTADO_PAGO"]=="No pagado"][["SUS_RAZON_SOCIAL","RFC_SUS","SUS_NOMBRE","MONTO_SUS"]].copy()
    if not no_pag.empty:
        no_pag["Costo + IVA"]=(no_pag["MONTO_SUS"].fillna(0)*(1+iva)).round(2)
        no_pag=no_pag.rename(columns={"SUS_RAZON_SOCIAL":"Razón social","RFC_SUS":"RFC","SUS_NOMBRE":"Cuenta","MONTO_SUS":"Costo"})
        no_pag=no_pag[["Razón social","RFC","Cuenta","Costo","Costo + IVA"]].sort_values(["Razón social","Cuenta"], na_position="last")
    else:
        no_pag=pd.DataFrame(columns=["Razón social","RFC","Cuenta","Costo","Costo + IVA"])

    # 2) PAGADAS (con Fecha factura)
    pag=det[det["ESTADO_PAGO"]=="Pagado"][["SUS_RAZON_SOCIAL","RFC_SUS","SUS_NOMBRE","MONTO_SUS","FACTURA_FECHA"]].copy()
    if not pag.empty:
        pag["Costo + IVA"]=(pag["MONTO_SUS"].fillna(0)*(1+iva)).round(2)
        pag=pag.rename(columns={
            "SUS_RAZON_SOCIAL":"Razón social","RFC_SUS":"RFC","SUS_NOMBRE":"Cuenta",
            "MONTO_SUS":"Costo","FACTURA_FECHA":"Fecha factura"
        })
        pag=pag[["Razón social","RFC","Cuenta","Costo","Costo + IVA","Fecha factura"]].sort_values(["Razón social","Cuenta"], na_position="last")
    else:
        pag=pd.DataFrame(columns=["Razón social","RFC","Cuenta","Costo","Costo + IVA","Fecha factura"])

    # 3) FACTURAS NO ASIGNADAS (Razón social, RFC, Folio, Monto factura, Fecha)
    usados=set(det["FACTURA_NUMERO"].dropna().astype(str).tolist()) if "FACTURA_NUMERO" in det.columns else set()
    f_no=facturas_df.copy()
    if "NUMERO" not in f_no.columns and "Número" in f_no.columns: f_no=f_no.rename(columns={"Número":"NUMERO"})
    if "NUMERO" not in f_no.columns and "Numero" in f_no.columns: f_no=f_no.rename(columns={"Numero":"NUMERO"})
    if "NUMERO" in f_no.columns:
        f_no["NUMERO_STR"]=f_no["NUMERO"].astype(str)
        if usados: f_no=f_no[~f_no["NUMERO_STR"].isin(usados)]
        f_no=f_no.drop(columns=["NUMERO_STR"], errors="ignore")

    razon_col=next((c for c in ["Razón social","RAZÓN SOCIAL","Razon social"] if c in f_no.columns), None)
    rfc_col="RFC" if "RFC" in f_no.columns else ("RFC_FACTURA" if "RFC_FACTURA" in f_no.columns else None)
    monto_col="MONTO_FACTURA_R" if "MONTO_FACTURA_R" in f_no.columns else ("MONTO_FACTURA" if "MONTO_FACTURA" in f_no.columns else None)
    fecha_col="Fecha" if "Fecha" in f_no.columns else None
    folio_col="Folio" if "Folio" in f_no.columns else None
    cols_order=[c for c in [razon_col, rfc_col, folio_col, monto_col, fecha_col] if c]
    if cols_order:
        f_no_listo=f_no[cols_order].copy()
        ren={}
        if razon_col: ren[razon_col]="Razón social"
        if rfc_col: ren[rfc_col]="RFC"
        if folio_col: ren[folio_col]="Folio"
        if monto_col: ren[monto_col]="Monto factura"
        # aquí el nombre debe ser exactamente "Fecha" (no "Fecha factura")
        if fecha_col: ren[fecha_col]="Fecha"
        f_no_listo=f_no_listo.rename(columns=ren)
    else:
        f_no_listo=pd.DataFrame(columns=["Razón social","RFC","Folio","Monto factura","Fecha"])

    # Hoja de detalle para auditoría
    detalle_vis = det.copy()
    detalle_vis["Costo + IVA"] = (detalle_vis["MONTO_SUS"].fillna(0)*(1+iva)).round(2)

    # Excel en memoria con el layout exacto
    bio=BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter", datetime_format="dd/mm/yyyy", date_format="dd/mm/yyyy") as writer:
        sheet="1_Listado"
        ws=writer.book.add_worksheet(sheet); writer.sheets[sheet]=ws

        # 1) NO PAGADAS
        ws.write(0,0,"NO PAGADAS")
        no_pag.to_excel(writer, sheet_name=sheet, index=False, startrow=2)
        header_fmt=writer.book.add_format({"bold":True}); ws.set_row(2,None,header_fmt)

        # 2) PAGADAS
        startrow_pag=len(no_pag)+5
        ws.write(startrow_pag,0,"PAGADAS")
        pag.to_excel(writer, sheet_name=sheet, index=False, startrow=startrow_pag+2)
        ws.set_row(startrow_pag+2,None,header_fmt)

        # 3) FACTURAS NO ASIGNADAS
        startrow_noasig=startrow_pag+2+len(pag)+3
        ws.write(startrow_noasig,0,"FACTURAS NO ASIGNADAS")
        f_no_listo.to_excel(writer, sheet_name=sheet, index=False, startrow=startrow_noasig+2)
        ws.set_row(startrow_noasig+2,None,header_fmt)

        # Ancho de columnas
        ws.set_column(0,0,40); ws.set_column(1,1,16); ws.set_column(2,2,36); ws.set_column(3,3,18); ws.set_column(4,4,16)

        # Hoja de apoyo
        detalle_vis.to_excel(writer, index=False, sheet_name="detalle_Completo")

    bio.seek(0)
    return bio.getvalue(), (no_pag, pag, f_no_listo)

# ---------------------- UI ----------------------
st.title("Cruce de Facturas vs Suscriptores")
st.caption("Filtro previo obligatorio: **SUS_OFICINA = CDMX** y **ESTATUS = ACTIVO**. Además, solo **SUS_GIRO = SI**. Formato de salida fijo.")

with st.sidebar:
    st.header("Parámetros")
    hoja_sus = st.text_input("Hoja de suscriptores", value="Completo")
    iva = st.number_input("IVA", value=0.16, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
    tol = st.number_input("Tolerancia (MXN)", value=1.0, min_value=0.0, step=0.5, format="%.2f")
    st.caption("Protege la app con `APP_PASSWORD` en Secrets o variables de entorno.")

c1, c2 = st.columns(2)
with c1:
    sus_file = st.file_uploader("Excel de suscriptores", type=["xlsx","xls"])
with c2:
    fact_file = st.file_uploader("Excel de facturas", type=["xlsx","xls"])

if sus_file and fact_file:
    try:
        sus_xls = pd.ExcelFile(sus_file)
        if hoja_sus in sus_xls.sheet_names:
            sus_raw = pd.read_excel(sus_xls, sheet_name=hoja_sus)
        else:
            st.warning(f"No se encontró la hoja '{hoja_sus}'. Se usará: {sus_xls.sheet_names[0]}")
            sus_raw = pd.read_excel(sus_xls, sheet_name=0)
        fact_raw = pd.read_excel(fact_file)

        fact_df = prep_facturas(fact_raw)
        sus_df = prep_suscriptores(sus_raw, hoja_sus, oficina="CDMX", estatus="ACTIVO")

        with st.spinner("Procesando cruce..."):
            detalle = asignar_matches(sus_df, fact_df, iva=iva, tol=tol)
            excel_bytes, (no_pag_tbl, pag_tbl, no_asign_tbl) = build_excel(detalle, fact_df, iva=iva)

        st.success("¡Cruce generado!")
        st.download_button(
            label="Descargar Excel",
            data=excel_bytes,
            file_name=f"resultado_cruce_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.subheader("Vista previa")
        st.markdown("**NO PAGADAS**")
        st.dataframe(no_pag_tbl, use_container_width=True, hide_index=True)
        st.markdown("**PAGADAS**")
        st.dataframe(pag_tbl, use_container_width=True, hide_index=True)
        st.markdown("**FACTURAS NO ASIGNADAS**")
        st.dataframe(no_asign_tbl, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
else:
    st.info("Carga los dos archivos para comenzar.")
