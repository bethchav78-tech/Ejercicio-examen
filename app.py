import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Urgencias Hospitalarias - Simulación", layout="wide")

st.title("Simulación de Sistema de Urgencias Hospitalarias")
st.markdown("Modelo teórico M/M/1 + simulación de pacientes para análisis operativo.")

# -----------------------------
# Funciones teóricas
# -----------------------------
def mm1_metrics(lmbda, mu):
    if lmbda <= 0 or mu <= 0:
        return None
    if lmbda >= mu:
        return {
            "stable": False,
            "rho": lmbda / mu
        }

    rho = lmbda / mu
    p0 = 1 - rho
    lq = (lmbda ** 2) / (mu * (mu - lmbda))
    l = lmbda / (mu - lmbda)
    wq = lq / lmbda
    w = 1 / (mu - lmbda)

    return {
        "stable": True,
        "rho": rho,
        "p0": p0,
        "lq": lq,
        "l": l,
        "wq": wq,
        "w": w
    }


def prob_n_or_more(rho, n):
    return rho ** n


def prob_wait_more_than_t(lmbda, mu, t_hours):
    rho = lmbda / mu
    return rho * math.exp(-(mu - lmbda) * t_hours)


# -----------------------------
# Simulación
# -----------------------------
def simulate_mm1(lmbda, mu, n_customers, seed=42):
    np.random.seed(seed)

    interarrival_times = np.random.exponential(scale=1/lmbda, size=n_customers)
    service_times = np.random.exponential(scale=1/mu, size=n_customers)

    arrival_times = np.cumsum(interarrival_times)

    service_start = np.zeros(n_customers)
    service_end = np.zeros(n_customers)
    wait_times = np.zeros(n_customers)
    system_times = np.zeros(n_customers)

    for i in range(n_customers):
        if i == 0:
            service_start[i] = arrival_times[i]
        else:
            service_start[i] = max(arrival_times[i], service_end[i-1])

        wait_times[i] = service_start[i] - arrival_times[i]
        service_end[i] = service_start[i] + service_times[i]
        system_times[i] = service_end[i] - arrival_times[i]

    df = pd.DataFrame({
        "Paciente": np.arange(1, n_customers + 1),
        "Tiempo_entre_llegadas_h": interarrival_times,
        "Hora_llegada_h": arrival_times,
        "Tiempo_servicio_h": service_times,
        "Inicio_servicio_h": service_start,
        "Fin_servicio_h": service_end,
        "Espera_cola_h": wait_times,
        "Tiempo_total_sistema_h": system_times
    })

    return df


# -----------------------------
# Inputs
# -----------------------------
st.sidebar.header("Parámetros del sistema")
lmbda = st.sidebar.number_input("Tasa de llegada λ (pacientes/hora)", min_value=0.1, value=10.0, step=0.1)
mu = st.sidebar.number_input("Tasa de servicio μ (pacientes/hora)", min_value=0.1, value=14.0, step=0.1)
threshold_minutes = st.sidebar.number_input("Umbral crítico de espera (min)", min_value=1.0, value=12.0, step=1.0)
n_value = st.sidebar.number_input("Valor n para P(N ≥ n)", min_value=1, value=5, step=1)
n_sim = st.sidebar.number_input("Número de pacientes a simular", min_value=10, value=200, step=10)
seed = st.sidebar.number_input("Semilla aleatoria", min_value=0, value=42, step=1)

metrics = mm1_metrics(lmbda, mu)

if metrics is None:
    st.error("Los parámetros deben ser positivos.")
    st.stop()

if not metrics["stable"]:
    st.error("El sistema no es estable porque λ ≥ μ. No se puede analizar correctamente con M/M/1 estable.")
    st.write(f"Utilización actual: {metrics['rho']:.4f}")
    st.stop()

# -----------------------------
# Resultados teóricos
# -----------------------------
st.header("Resultados teóricos M/M/1")

rho = metrics["rho"]
p0 = metrics["p0"]
lq = metrics["lq"]
l = metrics["l"]
wq = metrics["wq"]
w = metrics["w"]

prob_n = prob_n_or_more(rho, n_value)
prob_wait = prob_wait_more_than_t(lmbda, mu, threshold_minutes / 60)

col1, col2, col3 = st.columns(3)
col1.metric("Utilización ρ", f"{rho:.4f}")
col2.metric("P₀", f"{p0:.4f}")
col3.metric("Lq", f"{lq:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("L", f"{l:.4f}")
col5.metric("Wq (min)", f"{wq * 60:.2f}")
col6.metric("W (min)", f"{w * 60:.2f}")

st.subheader("Probabilidades de riesgo")
st.write(f"**P(N ≥ {n_value}) = {prob_n:.4f}**")
st.write(f"**P(Wq > {threshold_minutes:.0f} min) = {prob_wait:.4f}**")

# -----------------------------
# Interpretación automática
# -----------------------------
st.header("Interpretación automática")

if rho < 0.6:
    st.success("El sistema tiene una carga relativamente cómoda. El riesgo de congestión es bajo a moderado.")
elif rho < 0.8:
    st.warning("El sistema opera con carga importante. Puede funcionar, pero existe riesgo operativo en horas pico.")
else:
    st.error("El sistema está muy exigido. La probabilidad de congestión y espera crítica es alta.")

if prob_wait > 0.30:
    st.error("La probabilidad de espera crítica es alta para un entorno hospitalario.")
elif prob_wait > 0.15:
    st.warning("La probabilidad de espera crítica es moderada y debe vigilarse.")
else:
    st.success("La probabilidad de espera crítica es relativamente baja.")

# -----------------------------
# Distribución de estados
# -----------------------------
st.header("Distribución de probabilidad de estados")
st.markdown("Probabilidad de tener n pacientes en el sistema.")

n_states = np.arange(0, 15)
pn = [(1 - rho) * (rho ** n) for n in n_states]

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(n_states, pn)
ax1.set_xlabel("Número de pacientes en el sistema (n)")
ax1.set_ylabel("P(n)")
ax1.set_title("Distribución de estados del sistema")
st.pyplot(fig1)

# -----------------------------
# Simulación
# -----------------------------
st.header("Simulación de pacientes")
df = simulate_mm1(lmbda, mu, int(n_sim), int(seed))

avg_wait_sim = df["Espera_cola_h"].mean() * 60
avg_system_sim = df["Tiempo_total_sistema_h"].mean() * 60
critical_pct = (df["Espera_cola_h"] * 60 > threshold_minutes).mean() * 100

col7, col8, col9 = st.columns(3)
col7.metric("Espera promedio simulada (min)", f"{avg_wait_sim:.2f}")
col8.metric("Tiempo total simulado (min)", f"{avg_system_sim:.2f}")
col9.metric("% pacientes con espera crítica", f"{critical_pct:.2f}%")

st.dataframe(df.head(30), use_container_width=True)

# -----------------------------
# Gráficos de simulación
# -----------------------------
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["Paciente"], df["Espera_cola_h"] * 60)
ax2.set_xlabel("Paciente")
ax2.set_ylabel("Espera en cola (min)")
ax2.set_title("Tiempo de espera por paciente")
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.hist(df["Tiempo_total_sistema_h"] * 60, bins=20)
ax3.set_xlabel("Tiempo total en sistema (min)")
ax3.set_ylabel("Frecuencia")
ax3.set_title("Distribución del tiempo total en el sistema")
st.pyplot(fig3)

# -----------------------------
# Comparación teórico vs simulado
# -----------------------------
st.header("Comparación teórico vs simulado")
comparison = pd.DataFrame({
    "Métrica": ["Wq (min)", "W (min)"],
    "Teórico": [wq * 60, w * 60],
    "Simulado": [avg_wait_sim, avg_system_sim]
})
st.dataframe(comparison, use_container_width=True)

# -----------------------------
# Recomendación final
# -----------------------------
st.header("Recomendación automática")

if rho >= 0.8 or prob_wait >= 0.30:
    st.error(
        "Se recomienda aumentar capacidad de atención o rediseñar el flujo de urgencias. "
        "El sistema presenta riesgo operativo elevado."
    )
elif rho >= 0.65:
    st.warning(
        "El sistema es funcional pero exigido. Conviene monitorear horas pico y evaluar mejora preventiva."
    )
else:
    st.success(
        "El sistema opera en condiciones relativamente aceptables, aunque siempre conviene monitorear la variabilidad real."
    )
