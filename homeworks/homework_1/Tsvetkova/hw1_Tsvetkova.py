import matplotlib
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp

import numpy as np
from scipy.integrate import solve_ivp

import json

def calc_ws(gamma_wat: float) -> float:
    """
    Функция для расчета солесодержания в воде

    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.

    :return: солесодержание в воде, г/г
    """
    
    ws = (
            1 / (gamma_wat * 1000)
            * (1.36545 * gamma_wat * 1000 - (3838.77 * gamma_wat * 1000 - 2.009 * (gamma_wat * 1000) ** 2) ** 0.5)
    )
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if ws > 0:
        return ws
    else:
        return 0

def calc_rho_w(ws: float, t: float) -> float:
    """
    Функция для расчета плотности воды в зависимости от температуры и солесодержания

    :param ws: солесодержание воды, г/г
    :param t: температура, К

    :return: плотность воды, кг/м3
    """
    rho_w = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)

    return rho_w / (1 + (t - 273) * 1e-4 * (0.269 * (t - 273) ** 0.637 - 0.8))

def calc_mu_w(ws: float, t: float, p: float) -> float:
    """
    Функция для расчета динамической вязкости воды по корреляции Matthews & Russel

    :param ws: солесодержание воды, г/г
    :param t: температура, К
    :param p: давление, Па

    :return: динамическая вязкость воды, сПз
    """
    a = (
            109.574
            - (0.840564 * 1000 * ws)
            + (3.13314 * 1000 * ws ** 2)
            + (8.72213 * 1000 * ws ** 3)
    )
    b = (
            1.12166
            - 2.63951 * ws
            + 6.79461 * ws ** 2
            + 54.7119 * ws ** 3
            - 155.586 * ws ** 4
    )

    mu_w = (
            a * (1.8 * t - 460) ** (-b)
            * (0.9994 + 0.0058 * (p * 1e-6) + 0.6534 * 1e-4 * (p * 1e-6) ** 2)
    )
    return mu_w

def calc_n_re(rho_w: float, q_ms: float, mu_w: float, d_tub: float) -> float:
    """
    Функция для расчета числа Рейнольдса

    :param rho_w: плотность воды, кг/м3
    :param q_ms: дебит жидкости, м3/с
    :param mu_w: динамическая вязкость воды, сПз
    :param d_tub: диаметр НКТ, м

    :return: число Рейнольдса, безразмерн.
    """
    v = q_ms / (np.pi * d_tub ** 2 / 4)
    return rho_w * v * d_tub / mu_w * 1000

def calc_ff_churchill(n_re: float, roughness: float, d_tub: float) -> float:
    """
    Функция для расчета коэффициента трения по корреляции Churchill

    :param n_re: число Рейнольдса, безразмерн.
    :param roughness: шероховатость стен трубы, м
    :param d_tub: диаметр НКТ, м

    :return: коэффициент трения, безразмерн.
    """
    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16

    ff = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1/12)
    return ff

def calc_ff_jain(n_re: float, roughness: float, d_tub: float) -> float:
    """
    Функция для расчета коэффициента трения по корреляции Jain

    :param n_re: число Рейнольдса, безразмерн.
    :param roughness: шероховатость стен трубы, м
    :param d_tub: диаметр НКТ, м

    :return: коэффициент трения, безразмерн.
    """
    if n_re < 3000:
        ff = 64 / n_re
    else:
        ff = 1 / (1.14 - 2 * np.log10(roughness / d_tub + 21.25 / (n_re**0.9))) ** 2
    return ff

def calc_dp_dl_grav(rho_w: float, angle: float):
    """
    Функция для расчета градиента на гравитацию

    :param rho_w: плотность воды, кг/м3
    :param angle: угол наклона скважины к горизонтали, градусы

    :return: градиент давления на гравитацию в трубе, Па/м
    """
    dp_dl_grav = rho_w * 9.81 * np.sin(angle / 180 * np.pi)
    return dp_dl_grav

def calc_dp_dl_fric(rho_w: float, mu_w: float, q_ms: float, d_tub: float, roughness: float):
    """
    Функция для расчета градиента давления на трение

    :param rho_w: плотность воды, кг/м3
    :param mu_w: динамическая вязкость воды, сПз
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: градиент давления в трубе, Па/м
    """
    if q_ms != 0:
        n_re = calc_n_re(rho_w, q_ms, mu_w, d_tub)
        ff = calc_ff_churchill(n_re, roughness, d_tub)
        dp_dl_fric = ff * rho_w * q_ms ** 2 / d_tub ** 5
    else:
        dp_dl_fric = 0
    return dp_dl_fric

def calc_dp_dl(rho_w: float, mu_w: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> float:
    """
    Функция для расчета градиента давления в трубе

    :param rho_w: плотность воды, кг/м3
    :param mu_w: динамическая вязкость воды, сПз
    :param angle: угол наклона скважины к горизонтали, градусы
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: градиент давления в трубе, Па/м
    """
    dp_dl_grav = calc_dp_dl_grav(rho_w, angle)
    dp_dl_fric = calc_dp_dl_fric(rho_w, mu_w, q_ms, d_tub, roughness)
    dp_dl = dp_dl_grav - 0.815 * dp_dl_fric

    return dp_dl

plt.figure(figsize=(10, 10))

plt.subplot(3, 3, 1)
x = np.linspace(0, 100, 10)                          
plt.plot(x, [calc_rho_w(0, t+273) for t in x])
plt.title('Зависимость плотности от температуры', fontsize=5)

plt.subplot(3, 3, 2)
x = np.linspace(992, 1300, 50)                          
plt.plot(x, [calc_ws(gamma_wat/1000) for gamma_wat in x])
plt.title('Зависимость солености от плотности воды', fontsize=5)

plt.subplot(3, 3, 3)
x = np.linspace(0, 100, 50)
plt.plot(x, [calc_mu_w(0.0001, t+273, 1*101325) for t in x], label = "Salinity 0.0001")
plt.plot(x, [calc_mu_w(0.001, t+273, 1*101325) for t in x], label = "Salinity 0.001")
plt.plot(x, [calc_mu_w(0.01, t+273, 1*101325) for t in x], label = "Salinity 0.01")
plt.plot(x, [calc_mu_w(0.1, t+273, 1*101325) for t in x], label = "Salinity 0.1")
plt.title('Зависимость вязкости от температуры', fontsize=5)
plt.xlabel("Температура, С", fontsize=5)
plt.ylabel("Динамическая вязкость, СП", fontsize=5)
plt.legend()

plt.subplot(3, 3, 4)
x = np.linspace(0, 5, 50)
plt.plot(x, [calc_n_re(rho_w=1000, q_ms=t/86400, mu_w=1, d_tub=0.062) for t in x])
plt.title('Зависимость числа Рейнольдса от дебита нагнетательной скважины', fontsize=5)
plt.xlabel("Дебит м3/сут", fontsize=5)

plt.subplot(3, 3, 5)
x = np.linspace(1, 50, 30)
n_re_list = [calc_n_re(rho_w=1000, q_ms=t/86400, mu_w=1, d_tub=0.062) for t in x]

plt.plot(x, [calc_ff_churchill(t, 0.0001, 0.62) for t in n_re_list], label ="Расчет по Джейн")
plt.plot(x, [calc_ff_jain(t, 0.0001, 0.62) for t in n_re_list], label ="Расчет по Черчилю")
plt.title('Зависимость коэффициента трения от дебита нагнетательной скважины', fontsize=5)
plt.xlabel("Дебит жидкости, м3/сут", fontsize=5)
plt.ylabel("коэффициент трения", fontsize=5)
plt.legend()

plt.subplot(3, 3, 6)
x = np.linspace(1, 5000, 30)
plt.plot(x, [calc_dp_dl(rho_w=1000, mu_w=1, angle=90, q_ms=t/86400, d_tub=0.062, roughness=0.001) for t in x])
plt.title('Зависимость градиента давления от дебита', fontsize=5)

plt.subplot(3, 3, 7)
x = np.linspace(1, 5000, 30)
plt.plot(x, [calc_dp_dl_grav(rho_w=1000, angle=90) for t in x])

plt.subplot(3, 3, 8)
x = np.linspace(1, 5000, 30)
plt.plot(x, [calc_dp_dl_fric(rho_w=1000, mu_w=1, q_ms=t/86400, d_tub=0.062, roughness=0.001) for t in x])
plt.title('Зависимость градиента давления по трению от дебита', fontsize=5 )
plt.show()

Q = 80 # дебит флюида, м3/cут
q_m3_sec = Q / 86400 # дебит флюида, м3/c
gamma_water = 1.0032771912327783 # относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм
H = 2001.8091648807663 # измеренная глубина забоя скважины
d_tub = 0.06360710809082407 # диаметр НКТ, м
angle = 62.59144715331712 # угол наклона скважины к горизонтали, градусы
roughness = 0.00014447597923425548 # шероховатость трубы, м
p_wh = 164.07965337809787 * 101325# давление на устье, Па
t_wh = 32.86207597525522 # температура на устье скважины, С
temp_grad = 2.0668596326592388 # геотермический градиент, К/м * (1e-2)

def __integr_func(h: float, pt: tuple, temp_grad: float, gamma_wat: float,angle: float, q_ms: float, d_tub: float, roughness: float) -> tuple:
    """
    Функция для интегрирования трубы

    :param h: текущая глубина, м
    :param pt: текущее давление, Па и текущая температура, К
    :param temp_grad: геотермический градиент, К/м * (1e-2)
    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
    :param angle: угол наклона скважины к горизонтали, градусы
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: градиенты давления, Па/м и температуры, К/м
    """
    p, t = pt
    ws = calc_ws(gamma_wat)
    ro_wat = calc_rho_w(ws, t)
    mu_wat = calc_mu_w(ws, t, p)

    dt_dl = temp_grad * 1e-2
    dp_dl = calc_dp_dl(ro_wat, mu_wat, angle, q_ms, d_tub, roughness)
    return [dp_dl, dt_dl]

def calc_pipe(p_wh: float, t_wh: float, h0: float, md_vdp: float, temp_grad: float, gamma_wat: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> tuple:
    """
    Функция для расчета давления в трубе

    :param p_wh: буферное давление, Па
    :param t_wh: температура жидкости у буферной задвижки, К
    :param h0: начальная глубина, м
    :param md_vdp: глубина верхних дыр перфорации, м
    :param temp_grad: геотермический градиент, К/м * (1e-2)
    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
    :param angle: угол наклона скважины к горизонтали, градусы
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: давление, Па и температура, K, глубины
    """
    h = np.linspace(h0, md_vdp, 100)
    
    res = solve_ivp(
        __integr_func,
        t_span=(h0, md_vdp),
        y0=[p_wh, t_wh],
        t_eval=h,
        args=(temp_grad, gamma_wat, angle, q_ms, d_tub, roughness),
        method="RK45",
        dense_output=True,
    )

    p_res = res.y[0]
    t_res = res.y[1]
    return p_res, t_res, h
        
def calc_p_wf(p_wh: float, t_wh: float, h0: float, md_vdp: float, temp_grad: float, gamma_wat: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> float:
    """
    Функция для расчета давления на забое скважины

    :param p_wh: буферное давление, Па
    :param t_wh: температура жидкости у буферной задвижки, К
    :param h0: начальная глубина, м
    :param md_vdp: глубина верхних дыр перфорации, м
    :param temp_grad: геотермический градиент, К/м * (1e-2)
    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
    :param angle: угол наклона скважины к горизонтали, градусы
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: давление на забое скважины, Па
    """
    p_res, t_res, h = calc_pipe(p_wh, t_wh, h0, md_vdp, temp_grad, gamma_wat, angle, q_ms, d_tub, roughness)
    return p_res[-1]


results = calc_pipe(p_wh, t_wh+273, h0=0, md_vdp=H, temp_grad=temp_grad, gamma_wat=gamma_water, angle=angle, q_ms=q_m3_sec, d_tub=d_tub, roughness=roughness)
p_res = results[0] / 101325
t_res = results[1] - 273
h_res = results[2]

plt.subplot(2, 2, 1)
plt.plot(p_res, h_res, label ="давление")
plt.plot(t_res, h_res, label ="температура")
plt.xlabel("P, Т")
plt.ylabel("h, длина скважины, м")
plt.grid()
ax = plt.gca()
ax.invert_yaxis()
plt.legend()
plt.title("Распределение давления")

# VLP
plt.subplot(2, 2, 2) 
p_wf_r = []
q = list(range(0, 401, 10))

for i in q:
    q_m3_sec = i / 86400
    p_wf = calc_p_wf(
        p_wh=p_wh,
        t_wh=t_wh + 273,
        h0=0,
        md_vdp=H,
        temp_grad=temp_grad,
        gamma_wat=gamma_water,
        angle=angle,
        q_ms=q_m3_sec,
        d_tub=d_tub,
        roughness=roughness,
    )
    p_wf_atm = p_wf / 101325
    p_wf_r.append(p_wf_atm)

plt.plot(q, p_wf_r)
plt.xlabel("Дебит закачиваемой жидкости, м3/сут", )
plt.ylabel("Забойное давление, атм")
plt.title("VLP")
plt.grid()
plt.show()

result = {"q_liq": q, "p_wf": p_wf_r}
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)