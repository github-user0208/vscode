"""
AI에게 태양계 설계를 맡긴다면?
N-body 시뮬레이션과 유전 알고리즘을 이용한 가상 행성계 최적화 설계

필요 라이브러리: numpy, matplotlib

단위계
------
계산이 다루기 쉽도록 천문단위계를 사용한다.
    거리  : AU  (지구와 태양 사이 거리 = 1)
    질량  : 태양질량 (M_sun = 1)
    시간  : 년(yr)
이 단위계에서는 만유인력 상수가 G = 4*pi**2 이 되어,
반지름 1AU·질량 1M_sun 인 별 주위를 도는 행성의 공전주기가 정확히 1년이 된다.

구성
----
1. N-body 시뮬레이션 (velocity-Verlet / leapfrog 수치적분)
2. 안정성 평가 (이탈, 근접 충돌, 궤도 반경 변동)
3. 유전 알고리즘 (안정성 최대화 / 행성 수 최대화 / 생명가능영역 행성 수 최대화)
4. 결과 시각화
5. 실제 태양계와의 비교
"""

import copy
import platform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

G = 4 * np.pi ** 2  # AU, M_sun, yr 단위계에서의 중력 상수


def _set_korean_font():
    """그래프에 한글이 깨지지 않도록 운영체제에 맞는 한글 폰트를 자동으로 찾아 설정한다.
    (Windows: 맑은 고딕 / macOS: AppleGothic / Linux: 나눔고딕 등)
    설치된 한글 폰트가 없으면 기본 폰트를 그대로 사용한다(그 경우 한글이 네모로 보일 수 있음).
    """
    candidates = {
        'Windows': ['Malgun Gothic'],
        'Darwin': ['AppleGothic'],
    }.get(platform.system(), ['NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR', 'Noto Sans CJK JP'])

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


_set_korean_font()


# =====================================================================
# 1. N-body 시뮬레이션
# =====================================================================

def compute_accel(pos, masses):
    """모든 천체가 서로에게 받는 중력 가속도를 한 번에 계산한다 (완전 N-body).

    pos    : (n, 2) 각 천체의 위치
    masses : (n,)   각 천체의 질량
    반환   : (n, 2) 각 천체가 받는 가속도
    """
    diff = pos[None, :, :] - pos[:, None, :]        # diff[i, j] = pos[j] - pos[i]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)                      # 자기 자신과의 나눗셈 방지(임시값)
    inv_dist3 = 1.0 / dist ** 3
    np.fill_diagonal(inv_dist3, 0.0)                  # 자기 자신에 의한 힘은 0
    weight = G * masses[None, :] * inv_dist3          # weight[i, j]
    acc = np.einsum('ij,ijk->ik', weight, diff)
    return acc

'''
def simulate(masses, positions, velocities, t_max, dt,
             ejection_radius=60.0, collision_dist=0.03, capture_every=4):
    """leapfrog(velocity-Verlet) 방식으로 N-body 궤도를 수치적분한다.

    안정성 판단에 필요한 두 가지 사건을 시뮬레이션 도중 함께 감지한다.
        - 충돌(collision): 두 천체 사이 거리가 collision_dist보다 가까워짐
        - 이탈(ejection)  : 질량중심으로부터 ejection_radius보다 멀어짐
    두 사건 중 하나가 발생하면 이후 계산은 의미가 없으므로 조기 종료한다.
    """
    n = len(masses)
    pos, vel = positions.copy(), velocities.copy()
    steps = int(t_max / dt)

    traj = [pos.copy()]
    collided = False
    ejected = set()

    acc = compute_accel(pos, masses)
    step = 0
    for step in range(1, steps + 1):
        vel_half = vel + 0.5 * dt * acc
        pos = pos + dt * vel_half
        acc = compute_accel(pos, masses)
        vel = vel_half + 0.5 * dt * acc

        # --- 충돌 감지 ---
        dmat = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        if dmat.min() < collision_dist:
            collided = True

        # --- 이탈 감지 (질량중심 기준) ---
        com = (masses[:, None] * pos).sum(axis=0) / masses.sum()
        dist_from_com = np.linalg.norm(pos - com, axis=1)
        newly_ejected = set(np.where(dist_from_com > ejection_radius)[0]) - {0}
        ejected |= newly_ejected

        if step % capture_every == 0:
            traj.append(pos.copy())

        if collided or len(ejected) >= n - 1:
            break

    return {
        'traj': np.array(traj),      # (T, n, 2)
        'masses': masses,
        'collided': collided,
        'ejected': ejected,
        'steps_run': step,
    }
'''

def simulate(masses, positions, velocities, t_max, dt,
             ejection_radius=60.0, collision_dist=0.03, capture_every=4):
    """leapfrog(velocity-Verlet) 방식으로 N-body 궤도를 수치적분한다.

    안정성 판단에 필요한 두 가지 사건을 시뮬레이션 도중 함께 감지한다.
        - 충돌(collision): 두 천체 사이 거리가 collision_dist보다 가까워짐
        - 이탈(ejection)  : 질량중심으로부터 ejection_radius보다 멀어짐
    두 사건 중 하나가 발생하면 이후 계산은 의미가 없으므로 조기 종료한다.
    """
    n = len(masses)
    pos, vel = positions.copy(), velocities.copy()
    steps = int(t_max / dt)

    traj = [pos.copy()]
    vel_traj = [vel.copy()]     # 궤도요소(a, e)의 시간에 따른 변화를 보려면 속도 이력도 필요
    collided = False
    ejected = set()

    acc = compute_accel(pos, masses)
    step = 0
    for step in range(1, steps + 1):
        vel_half = vel + 0.5 * dt * acc
        pos = pos + dt * vel_half
        acc = compute_accel(pos, masses)
        vel = vel_half + 0.5 * dt * acc

        # --- 충돌 감지 ---
        dmat = np.linalg.norm(pos[None, :, :] - pos[:, None, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        if dmat.min() < collision_dist:
            collided = True

        # --- 이탈 감지 (질량중심 기준) ---
        com = (masses[:, None] * pos).sum(axis=0) / masses.sum()
        dist_from_com = np.linalg.norm(pos - com, axis=1)
        newly_ejected = set(np.where(dist_from_com > ejection_radius)[0]) - {0}
        ejected |= newly_ejected

        if step % capture_every == 0:
            traj.append(pos.copy())
            vel_traj.append(vel.copy())

        if collided or len(ejected) >= n - 1:
            break

    return {
        'traj': np.array(traj),          # (T, n, 2) 위치 이력
        'vel_traj': np.array(vel_traj),  # (T, n, 2) 속도 이력
        'masses': masses,
        'collided': collided,
        'ejected': ejected,
        'steps_run': step,
    }

# =====================================================================
# 2. 안정성 평가
# =====================================================================

def individual_to_state(planets, star_mass=1.0):
    """[행성 딕셔너리 리스트] -> (질량, 위치, 속도) 배열로 변환.

    각 행성은 별 주위 원 궤도에 가까운 초기 속도(v_factor로 살짝 변형)를 갖는다.
    계 전체의 운동량 합이 0이 되도록 별의 속도를 보정한다(질량중심 정지).
    """
    n = len(planets)
    masses = np.array([star_mass] + [p['m'] for p in planets])
    positions = np.zeros((n + 1, 2))
    velocities = np.zeros((n + 1, 2))

    for i, p in enumerate(planets):
        ang = p['angle']
        positions[i + 1] = [p['r'] * np.cos(ang), p['r'] * np.sin(ang)]
        v_circular = np.sqrt(G * star_mass / p['r'])       # 원궤도 속력 근사
        speed = v_circular * p['v_factor']
        velocities[i + 1] = speed * np.array([-np.sin(ang), np.cos(ang)])

    total_momentum = (masses[1:, None] * velocities[1:]).sum(axis=0)
    velocities[0] = -total_momentum / masses[0]
    return masses, positions, velocities

'''
def stability_score(planets, star_mass=1.0, t_max=45.0, dt=0.02):
    """행성계의 안정성을 0~100점으로 평가한다.

    감점 요소
        - 충돌 발생            : -55점
        - 행성 1개 이탈(방출)당 : -14점
        - 궤도 반경의 변동      : 최대 -35점
                                  (살아남은 행성들의 (r_max-r_min)/r_mean 평균)
    """
    masses, positions, velocities = individual_to_state(planets, star_mass)
    result = simulate(masses, positions, velocities, t_max, dt)

    score = 100.0
    if result['collided']:
        score -= 55.0
    score -= 14.0 * len(result['ejected'])

    traj = result['traj']
    star_pos = traj[:, 0, :]
    variations = []
    for i in range(1, len(planets) + 1):
        if (i - 1) in result['ejected']:
            continue
        r = np.linalg.norm(traj[:, i, :] - star_pos, axis=1)
        if r.mean() > 1e-6:
            variations.append((r.max() - r.min()) / r.mean())
    if variations:
        score -= min(35.0, float(np.mean(variations)) * 45.0)

    return max(0.0, min(100.0, score)), result
'''

def stability_score(planets, star_mass=1.0, t_max=45.0, dt=0.02):
    """행성계의 안정성을 0~100점으로 평가한다.

    감점 요소
        - 충돌 발생            : -55점
        - 행성 1개 이탈(방출)당 : -14점
        - 궤도의 '드리프트'     : 최대 -35점
              (이심률이 큰 고정 타원 궤도는 감점하지 않는다. 대신 시뮬레이션
               구간을 전반부/후반부로 나누어, 궤도의 찌그러진 정도(이심률에
               대응하는 값)와 장반경(궤도 중심)이 시간이 지나며 실제로
               변해가는지 - 즉 섭동으로 궤도가 벌어지고 있는지 - 만 감점한다.)
    """
    masses, positions, velocities = individual_to_state(planets, star_mass)
    result = simulate(masses, positions, velocities, t_max, dt)

    score = 100.0
    if result['collided']:
        score -= 55.0
    score -= 14.0 * len(result['ejected'])

    traj = result['traj']            # (T, n+1, 2)
    star_pos = traj[:, 0, :]
    T = len(traj)

    drift_penalties = []
    if T >= 6:                       # 전반부/후반부를 나눌 만큼 궤적이 충분할 때만 평가
        mid = T // 2
        for i in range(1, len(planets) + 1):
            if (i - 1) in result['ejected']:
                continue
            r = np.linalg.norm(traj[:, i, :] - star_pos, axis=1)
            r1, r2 = r[:mid], r[mid:]
            if r1.mean() < 1e-6 or r2.mean() < 1e-6:
                continue
            shape1 = (r1.max() - r1.min()) / r1.mean()      # 전반부 궤도 이심률 대응값
            shape2 = (r2.max() - r2.min()) / r2.mean()      # 후반부 궤도 이심률 대응값
            shape_drift = abs(shape2 - shape1)               # 이심률 자체가 변했는가
            center_drift = abs(r2.mean() - r1.mean()) / r1.mean()  # 장반경이 이동했는가
            drift_penalties.append(shape_drift + center_drift)

    if drift_penalties:
        score -= min(35.0, float(np.mean(drift_penalties)) * 60.0)

    return max(0.0, min(100.0, score)), result


# =====================================================================
# 3. 유전 알고리즘
# =====================================================================

HZ_INNER, HZ_OUTER = 0.9, 1.6   # 생명가능영역(액체 물이 존재 가능한 궤도 범위) 근사값[AU]


def random_planet(rng, r_min=0.35, r_max=6.0):
    return {
        'r': float(rng.uniform(r_min, r_max)),
        'm': float(10 ** rng.uniform(-6, -3)),     # 약 0.3 지구질량 ~ 1 목성질량
        'v_factor': float(rng.uniform(0.9, 1.1)),
        'angle': float(rng.uniform(0, 2 * np.pi)),
    }


def fix_spacing(planets, min_gap=0.25):
    """궤도가 서로 너무 가까이서 시작하지 않도록 최소 간격을 보장한다."""
    planets = sorted(planets, key=lambda p: p['r'])
    for i in range(1, len(planets)):
        if planets[i]['r'] - planets[i - 1]['r'] < min_gap:
            planets[i]['r'] = planets[i - 1]['r'] + min_gap
    return planets


def random_individual(rng, min_p=1, max_p=20):
    n = int(rng.integers(min_p, max_p + 1))
    return fix_spacing([random_planet(rng) for _ in range(n)])


def fitness(planets, objective, star_mass=1.0, t_max=45.0, dt=0.02):
    """목표(objective)에 따라 적합도를 계산한다.

    - 'stability'    : 안정성 점수 그대로
    - 'planet_count' : 안정성 점수 x (행성 수가 많을수록 커지는 가중치)
    - 'habitable'    : 안정성 점수 x (생명가능영역 안 행성 수가 많을수록 커지는 가중치)

    안정성 점수에 '곱하는' 형태이기 때문에, 불안정한(점수가 낮은) 계는
    행성이 아무리 많아도 높은 적합도를 받을 수 없다.
    """
    score, _ = stability_score(planets, star_mass, t_max, dt)
    n_planets = len(planets)

    if objective == 'stability':
        return score
    elif objective == 'planet_count':
        return score * (1 + 0.12 * n_planets)
    elif objective == 'habitable':
        n_hz = sum(1 for p in planets if HZ_INNER <= p['r'] <= HZ_OUTER)
        return score * (1 + 0.30 * n_hz)
    else:
        raise ValueError(f"알 수 없는 objective: {objective}")


def tournament_select(pop, fits, rng, k=3):
    idxs = rng.integers(0, len(pop), k)
    best = max(idxs, key=lambda i: fits[i])
    return copy.deepcopy(pop[best])


def crossover(parent1, parent2, rng, min_p=1, max_p=20):
    """두 부모의 행성들을 섞은 뒤 무작위로 일부를 골라 자식 행성계를 만든다."""
    pool = copy.deepcopy(parent1) + copy.deepcopy(parent2)
    rng.shuffle(pool)
    n_child = int(rng.integers(min_p, max_p + 1))
    child = pool[:max(min_p, min(n_child, len(pool)))]
    return fix_spacing(child)


def mutate(planets, rng, rate=0.25, min_p=1, max_p=20):
    planets = copy.deepcopy(planets)
    for p in planets:
        if rng.random() < rate:
            p['r'] = max(0.3, p['r'] * rng.uniform(0.85, 1.15))
        if rng.random() < rate:
            p['m'] = float(np.clip(p['m'] * rng.uniform(0.6, 1.6), 1e-7, 3e-3))
        if rng.random() < rate:
            p['v_factor'] = float(np.clip(p['v_factor'] * rng.uniform(0.95, 1.05), 0.8, 1.2))

    if rng.random() < 0.15 and len(planets) < max_p:
        planets.append(random_planet(rng))
    if rng.random() < 0.15 and len(planets) > min_p:
        planets.pop(int(rng.integers(0, len(planets))))

    return fix_spacing(planets)


def run_ga(objective='stability', pop_size=24, generations=25,
           star_mass=1.0, t_max=45.0, dt=0.02, seed=0, verbose=True):
    """유전 알고리즘 메인 루프. (최고 개체, 최고 적합도, 세대별 기록)을 반환한다."""
    rng = np.random.default_rng(seed)
    population = [random_individual(rng) for _ in range(pop_size)]

    history = []
    best_ever, best_fit = None, -1.0

    for gen in range(generations):
        fits = [fitness(ind, objective, star_mass, t_max, dt) for ind in population]
        gen_best_idx = int(np.argmax(fits))

        if fits[gen_best_idx] > best_fit:
            best_fit = fits[gen_best_idx]
            best_ever = copy.deepcopy(population[gen_best_idx])

        history.append({'gen': gen, 'best': fits[gen_best_idx], 'avg': float(np.mean(fits))})
        if verbose:
            print(f"  세대 {gen+1:>2}/{generations}  최고 {fits[gen_best_idx]:6.2f}  평균 {np.mean(fits):6.2f}")

        # 엘리트 보존 + 나머지는 선택-교배-변이로 생성
        new_pop = [copy.deepcopy(population[gen_best_idx])]
        while len(new_pop) < pop_size:
            parent1 = tournament_select(population, fits, rng)
            parent2 = tournament_select(population, fits, rng)
            child = crossover(parent1, parent2, rng)
            child = mutate(child, rng)
            new_pop.append(child)
        population = new_pop

    return best_ever, best_fit, history


# =====================================================================
# 4. 시각화
# =====================================================================

def plot_fitness_history(histories, save_path=None):
    plt.figure(figsize=(7, 4.5))
    for label, hist in histories.items():
        gens = [h['gen'] + 1 for h in hist]
        best = [h['best'] for h in hist]
        plt.plot(gens, best, marker='o', markersize=3, label=label)
    plt.xlabel('세대')
    plt.ylabel('적합도 (fitness)')
    plt.title('목표별 세대에 따른 최고 적합도 변화')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_system(planets, star_mass=1.0, t_max=30.0, dt=0.02, title='', save_path=None):
    masses, positions, velocities = individual_to_state(planets, star_mass)
    result = simulate(masses, positions, velocities, t_max, dt, capture_every=2)
    traj = result['traj']
    star_traj = traj[:, 0, :]

    plt.figure(figsize=(5.5, 5.5))
    for i in range(1, len(masses)):
        plt.plot(traj[:, i, 0] - star_traj[:, 0], traj[:, i, 1] - star_traj[:, 1], linewidth=1.3)
    plt.scatter([0], [0], marker='*', s=250, color='orange', zorder=5)
    plt.gca().set_aspect('equal')
    plt.xlabel('AU')
    plt.ylabel('AU')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# =====================================================================
# 5. 실제 태양계와의 비교
# =====================================================================

# 실제 태양계 각 행성의 평균 궤도반지름[AU]과 질량[태양질량 대비] (근삿값)
SOLAR_SYSTEM = [
    {'name': '수성', 'r': 0.39, 'm': 1.66e-7},
    {'name': '금성', 'r': 0.72, 'm': 2.45e-6},
    {'name': '지구', 'r': 1.00, 'm': 3.00e-6},
    {'name': '화성', 'r': 1.52, 'm': 3.23e-7},
    {'name': '목성', 'r': 5.20, 'm': 9.55e-4},
    {'name': '토성', 'r': 9.58, 'm': 2.86e-4},
    {'name': '천왕성', 'r': 19.20, 'm': 4.37e-5},
    {'name': '해왕성', 'r': 30.05, 'm': 5.15e-5},
]

'''
def real_solar_system(rng):
    return [{'r': p['r'], 'm': p['m'], 'v_factor': 1.0, 'angle': float(rng.uniform(0, 2 * np.pi))}
            for p in SOLAR_SYSTEM]
'''

# 실제 행성 궤도이심률 (표준 근사값)
ECCENTRICITY = {
    '수성': 0.206, '금성': 0.007, '지구': 0.017, '화성': 0.093,
    '목성': 0.048, '토성': 0.056, '천왕성': 0.047, '해왕성': 0.010,
}

def real_solar_system(rng):
    """실제 태양계 궤도를 '근일점에서 출발'하는 조건으로 구성한다.

    SOLAR_SYSTEM의 r은 궤도 장반경(a)이므로, 실제 이심률 e를 반영하려면
    근일점 r0 = a(1-e)에서, 근일점 속력 배율 v_factor = sqrt(1+e)로
    출발시켜야 한다 (개별 planet의 velocity가 항상 접선 방향이라는 가정과
    맞으려면 근일점·원일점에서 출발해야 하기 때문).
    """
    planets = []
    for p in SOLAR_SYSTEM:
        a = p['r']
        e = ECCENTRICITY[p['name']]
        r0 = a * (1 - e)
        v_factor = np.sqrt(1 + e)
        planets.append({'r': r0, 'm': p['m'], 'v_factor': v_factor,
                         'angle': float(rng.uniform(0, 2 * np.pi))})
    return planets

def compare_with_solar_system(best_planets, label, rng, t_max=45.0, dt=0.02):
    real_planets = real_solar_system(rng)
    real_score, _ = stability_score(real_planets, t_max=t_max, dt=dt)
    best_score, _ = stability_score(best_planets, t_max=t_max, dt=dt)
    print(f"[{label}]")
    print(f"   AI가 설계한 행성계 안정성 점수   : {best_score:5.1f} / 100  (행성 수 {len(best_planets)})")
    print(f"   실제 태양계 안정성 점수(동일 기준) : {real_score:5.1f} / 100  (행성 수 {len(SOLAR_SYSTEM)})")
    return best_score, real_score

def osculating_elements(rel_pos, rel_vel, GM):
    """어느 순간의 (별 기준 상대위치, 상대속도)만으로 케플러 궤도요소(a, e)를 구한다.

    비스-비바 방정식을 그대로 사용하며, 위치·속도가 원궤도 접선 방향이 아닌
    임의의 방향이어도 성립하는 일반식이다. '만약 이 순간부터 다른 행성의
    중력이 사라진다면 어떤 타원을 그리게 되는가'를 뜻하는 순간 궤도요소
    (osculating orbital elements)로, 섭동이 있는 실제 궤도를 순간순간
    근사하는 표준적인 방법이다.
    """
    r = np.linalg.norm(rel_pos)
    v2 = float(rel_vel @ rel_vel)
    eps = v2 / 2 - GM / r                                  # 비궤도 에너지
    a = -GM / (2 * eps)
    h = rel_pos[0] * rel_vel[1] - rel_pos[1] * rel_vel[0]   # 각운동량(2D 스칼라)
    e = np.sqrt(max(0.0, 1 + 2 * eps * h ** 2 / GM ** 2))
    return a, e

def print_orbital_elements(planets, star_mass=1.0):
    """planets 리스트의 원시 값(r, v_factor, m)으로부터 실제 궤도요소를 계산해 표로 출력한다.

    r(출발 반경)에서 별 주위 원궤도 속도의 v_factor배로 접선 방향 속도를 주고
    출발했다는 조건만으로, 뉴턴 역학을 이용해 다음을 역산할 수 있다.
        장반경   a = r / (2 - v_factor**2)
        이심률   e = |1 - v_factor**2|
        근일점   a(1-e),  원일점 a(1+e)
        공전주기 T = sqrt(a**3 / star_mass)   (G=4π² 단위계에서의 케플러 제3법칙)

    주의: 다른 행성의 중력에 의한 섭동은 무시한, t=0 시점 기준 2체(별-행성)
    근사 궤도요소다. 실제 시뮬레이션에서는 행성끼리 서로 끌어당기므로
    시간이 지나며 이 값에서 조금씩 벗어날 수 있다.
    """
    EARTH_MASS, JUPITER_MASS = 3.00e-6, 9.55e-4   # 지구/목성 질량 (태양질량 대비)

    cols = ['행성', '출발반경 r(AU)', '장반경 a(AU)', '이심률 e',
            '근일점(AU)', '원일점(AU)', '공전주기(yr)', '질량']
    widths = [4, 12, 10, 8, 10, 10, 11, 14]
    header = ' | '.join(c.rjust(w) for c, w in zip(cols, widths))
    print(header)
    print('-' * len(header))

    for idx, p in enumerate(planets, start=1):
        r0, v_factor, m = p['r'], p['v_factor'], p['m']
        a = r0 / (2 - v_factor ** 2)
        e = abs(1 - v_factor ** 2)
        peri, apo = a * (1 - e), a * (1 + e)
        period = np.sqrt(a ** 3 / star_mass)

        mj = m / JUPITER_MASS
        #mass_str = f"{mj:.2f} 목성질량" if mj >= 0.05 else f"{m / EARTH_MASS:.2f} 지구질량"
        mass_str = f"{m / EARTH_MASS:.2f} 지구질량"

        row = [str(idx), f"{r0:.3f}", f"{a:.3f}", f"{e:.3f}",
               f"{peri:.3f}", f"{apo:.3f}", f"{period:.2f}", mass_str]
        print(' | '.join(v.rjust(w) for v, w in zip(row, widths)))

def print_orbital_evolution(planets, star_mass=1.0, t_max=45.0, dt=0.02):
    """각 행성의 궤도요소를 '초기(t=0)'와 '시뮬레이션 종료 시점' 두 번 계산해 비교 출력한다.

    r, a, e는 시뮬레이션 내내 고정된 값이 아니다. 이 함수는 그 사실을 그대로
    보여준다 — 섭동이 거의 없는 행성은 두 값이 비슷하게 나오고(안정),
    다른 행성의 중력을 강하게 받은 행성은 장반경·이심률 자체가 크게
    달라진다(불안정 = 궤도가 실제로 변형되었다는 뜻).
    """
    GM = G * star_mass
    masses, positions, velocities = individual_to_state(planets, star_mass)
    result = simulate(masses, positions, velocities, t_max, dt)
    traj, vel_traj = result['traj'], result['vel_traj']

    cols = ['행성', '초기 a(AU)', '초기 e', '최종 a(AU)', '최종 e', '상태']
    widths = [4, 10, 8, 10, 8, 10]
    header = ' | '.join(c.rjust(w) for c, w in zip(cols, widths))
    print(header)
    print('-' * len(header))

    for i in range(1, len(planets) + 1):
        if (i - 1) in result['ejected']:
            status = '이탈'
        elif result['collided']:
            status = '충돌 발생'
        else:
            status = '생존'

        rel_pos0 = traj[0, i] - traj[0, 0]
        rel_vel0 = vel_traj[0, i] - vel_traj[0, 0]
        a0, e0 = osculating_elements(rel_pos0, rel_vel0, GM)

        rel_posF = traj[-1, i] - traj[-1, 0]
        rel_velF = vel_traj[-1, i] - vel_traj[-1, 0]
        aF, eF = osculating_elements(rel_posF, rel_velF, GM)

        row = [str(i), f"{a0:.3f}", f"{e0:.3f}", f"{aF:.3f}", f"{eF:.3f}", status]
        print(' | '.join(v.rjust(w) for v, w in zip(row, widths)))

# =====================================================================
# 6. 실행 예시
# =====================================================================

if __name__ == '__main__':
    OBJECTIVES = ['stability', 'planet_count', 'habitable']
    LABELS = {
        'stability': '안정성 최대화',
        'planet_count': '행성 수 최대화',
        'habitable': '생명가능영역 행성 수 최대화',
    }

    best_systems, histories = {}, {}

    for obj in OBJECTIVES:
        print(f"\n=== [{LABELS[obj]}] 목표로 유전 알고리즘 실행 ===")
        best, best_fit, hist = run_ga(objective=obj, pop_size=24, generations=25,
                                       seed=hash(obj) % 10_000, verbose=True)
        best_systems[obj] = best
        histories[obj] = hist
        print(f"  -> 최종 최고 적합도 {best_fit:.2f}  (행성 수 {len(best)})")

    # 세대별 적합도 변화 비교
    plot_fitness_history({LABELS[o]: histories[o] for o in OBJECTIVES},
                          save_path='fitness_history.png')

    # 목표별 최적 행성계의 궤도 시각화
    for obj in OBJECTIVES:
        plot_system(best_systems[obj], title=f'{LABELS[obj]} — 최적 행성계',
                    save_path=f'best_{obj}.png')

    # 실제 태양계와 비교
    print("\n=== 실제 태양계와의 안정성 비교 ===")
    rng = np.random.default_rng(42)
    for obj in OBJECTIVES:
        compare_with_solar_system(best_systems[obj], LABELS[obj], rng)

for obj in OBJECTIVES:
    print(f"\n[{LABELS[obj]}]")
    print_orbital_elements(best_systems[obj])