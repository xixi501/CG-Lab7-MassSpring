"""
计算机图形学实验七 - 质点弹簧模型 (Mass-Spring Model)
基于 Taichi 的布料模拟，实现三种数值积分方法对比
包含选做内容：剪切弹簧、弯曲弹簧、球体碰撞
"""

import taichi as ti
import taichi.math as tm

# ============================================================
# 初始化 Taichi
# ============================================================
ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=42)

# ============================================================
# 模拟参数
# ============================================================
N = 20                       # 布料网格大小 N×N
grid_size = 0.05             # 初始网格间距（米）
dt = 1e-3                    # 时间步长
gravity = tm.vec3(0.0, -9.8, 0.0)
mass = 0.1                   # 每个质点的质量

# 弹簧参数
ks_struct = 8000.0           # 结构弹簧劲度系数
ks_shear = 4000.0            # 剪切弹簧劲度系数
ks_bend = 2000.0             # 弯曲弹簧劲度系数
kd = 5.0                     # 阻尼系数

# 防爆参数
max_velocity = 50.0

# 隐式欧拉迭代次数
implicit_iters = 5

# 球体碰撞参数
sphere_center = tm.vec3(0.0, -0.3, 0.0)
sphere_radius = 0.25
# === 以下参数通过 GUI 控制，使用 Taichi field 保证 Kernel 可见 ===
kd_field = ti.field(dtype=ti.f32, shape=())
enable_collision_field = ti.field(dtype=ti.i32, shape=())
enable_shear_field = ti.field(dtype=ti.i32, shape=())
enable_bend_field = ti.field(dtype=ti.i32, shape=())

# ============================================================
# Taichi Fields
# ============================================================
num_particles = N * N

# 质点的位置、速度、受力
pos = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
vel = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
force = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)

# 上一帧位置和速度（用于隐式欧拉）
pos_prev = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
vel_prev = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)

# 是否为固定点（顶部两行固定）
pinned = ti.field(dtype=ti.i32, shape=num_particles)

# 弹簧数据
MAX_SPRINGS = N * N * 12     # 足够容纳所有类型的弹簧
num_struct_springs = ti.field(dtype=ti.i32, shape=())
num_shear_springs = ti.field(dtype=ti.i32, shape=())
num_bend_springs = ti.field(dtype=ti.i32, shape=())

spring_a = ti.field(dtype=ti.i32, shape=MAX_SPRINGS)
spring_b = ti.field(dtype=ti.i32, shape=MAX_SPRINGS)
spring_rest = ti.field(dtype=ti.f32, shape=MAX_SPRINGS)
spring_type = ti.field(dtype=ti.i32, shape=MAX_SPRINGS)  # 0=struct, 1=shear, 2=bend

# 渲染数据
cloth_vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
cloth_triangles = ti.field(dtype=ti.i32, shape=(N - 1) * (N - 1) * 6)

# 球体渲染（用粒子近似）
num_sphere_particles = 200
sphere_particles = ti.Vector.field(3, dtype=ti.f32, shape=num_sphere_particles)

# 弹簧边渲染
MAX_EDGES = MAX_SPRINGS
edge_vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_EDGES * 2)
num_edges_to_render = ti.field(dtype=ti.i32, shape=())

# 积分器类型: 0=Explicit, 1=Semi-Implicit, 2=Implicit
integrator_type = ti.field(dtype=ti.i32, shape=())
paused = ti.field(dtype=ti.i32, shape=())
reset_flag = ti.field(dtype=ti.i32, shape=())


# ============================================================
# Kernel 1: 初始化质点位置
# ============================================================
@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x = (j - (N - 1) / 2.0) * grid_size
        y = 0.0
        z = i * grid_size
        pos[idx] = tm.vec3(x, y, z)
        vel[idx] = tm.vec3(0.0)
        force[idx] = tm.vec3(0.0)
        pos_prev[idx] = tm.vec3(x, y, z)
        vel_prev[idx] = tm.vec3(0.0)
        # 顶部两行固定
        if i == 0 or i == 1:
            pinned[idx] = 1
        else:
            pinned[idx] = 0


# ============================================================
# Kernel 2: 初始化弹簧拓扑
# ============================================================
@ti.kernel
def init_springs():
    num_struct_springs[None] = 0
    num_shear_springs[None] = 0
    num_bend_springs[None] = 0

    # 结构弹簧 (Structural)
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i + 1 < N:
            nbr = (i + 1) * N + j
            s = ti.atomic_add(num_struct_springs[None], 1)
            spring_a[s] = idx
            spring_b[s] = nbr
            spring_rest[s] = grid_size
            spring_type[s] = 0
        if j + 1 < N:
            nbr = i * N + (j + 1)
            s = ti.atomic_add(num_struct_springs[None], 1)
            spring_a[s] = idx
            spring_b[s] = nbr
            spring_rest[s] = grid_size
            spring_type[s] = 0

    # 剪切弹簧 (Shear) - 对角线
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i + 1 < N and j + 1 < N:
            nbr = (i + 1) * N + (j + 1)
            s = ti.atomic_add(num_shear_springs[None], 1)
            offset = num_struct_springs[None]
            spring_a[offset + s] = idx
            spring_b[offset + s] = nbr
            spring_rest[offset + s] = grid_size * ti.sqrt(2.0)
            spring_type[offset + s] = 1
        if i + 1 < N and j - 1 >= 0:
            nbr = (i + 1) * N + (j - 1)
            s = ti.atomic_add(num_shear_springs[None], 1)
            offset = num_struct_springs[None]
            spring_a[offset + s] = idx
            spring_b[offset + s] = nbr
            spring_rest[offset + s] = grid_size * ti.sqrt(2.0)
            spring_type[offset + s] = 1

    # 弯曲弹簧 (Bending) - 隔一个质点
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i + 2 < N:
            nbr = (i + 2) * N + j
            s = ti.atomic_add(num_bend_springs[None], 1)
            offset = num_struct_springs[None] + num_shear_springs[None]
            spring_a[offset + s] = idx
            spring_b[offset + s] = nbr
            spring_rest[offset + s] = grid_size * 2.0
            spring_type[offset + s] = 2
        if j + 2 < N:
            nbr = i * N + (j + 2)
            s = ti.atomic_add(num_bend_springs[None], 1)
            offset = num_struct_springs[None] + num_shear_springs[None]
            spring_a[offset + s] = idx
            spring_b[offset + s] = nbr
            spring_rest[offset + s] = grid_size * 2.0
            spring_type[offset + s] = 2


# ============================================================
# Kernel 3: 初始化渲染索引
# ============================================================
@ti.kernel
def init_render_indices():
    tri = 0
    for i, j in ti.ndrange(N - 1, N - 1):
        a = i * N + j
        b = i * N + (j + 1)
        c = (i + 1) * N + j
        d = (i + 1) * N + (j + 1)
        cloth_triangles[tri + 0] = a
        cloth_triangles[tri + 1] = b
        cloth_triangles[tri + 2] = c
        cloth_triangles[tri + 3] = b
        cloth_triangles[tri + 4] = d
        cloth_triangles[tri + 5] = c
        tri += 6


# ============================================================
# Kernel 4: 初始化球体渲染粒子
# ============================================================
@ti.kernel
def init_sphere_particles():
    for i in range(num_sphere_particles):
        theta = ti.random() * 2.0 * tm.pi
        phi = ti.acos(2.0 * ti.random() - 1.0)
        sphere_particles[i] = sphere_center + sphere_radius * tm.vec3(
            tm.sin(phi) * tm.cos(theta),
            tm.cos(phi),
            tm.sin(phi) * tm.sin(theta)
        )


# ============================================================
# ti.func: 速度钳制（防爆）
# ============================================================
@ti.func
def clamp_velocity(v):
    speed = v.norm()
    result = v
    if speed > max_velocity:
        result = v * (max_velocity / speed)
    return result


# ============================================================
# ti.func: 计算弹簧力（使用 atomic_add）
# ============================================================
@ti.func
def compute_spring_forces(positions, ks_structural, ks_shear_val, ks_bend_val):
    # 重置受力（所有质点）
    for idx in range(num_particles):
        force[idx] = tm.vec3(0.0)

    # 结构弹簧
    for s in range(num_struct_springs[None]):
        a = spring_a[s]
        b = spring_b[s]
        delta = positions[b] - positions[a]
        dist = delta.norm()
        if dist > 1e-8:
            direction = delta / dist
            f_mag = ks_structural * (dist - spring_rest[s])
            f = f_mag * direction
            ti.atomic_add(force[a][0], f[0])
            ti.atomic_add(force[a][1], f[1])
            ti.atomic_add(force[a][2], f[2])
            ti.atomic_add(force[b][0], -f[0])
            ti.atomic_add(force[b][1], -f[1])
            ti.atomic_add(force[b][2], -f[2])

    # 剪切弹簧
    if enable_shear_field[None] == 1:
        offset = num_struct_springs[None]
        for s in range(num_shear_springs[None]):
            a = spring_a[offset + s]
            b = spring_b[offset + s]
            delta = positions[b] - positions[a]
            dist = delta.norm()
            if dist > 1e-8:
                direction = delta / dist
                f_mag = ks_shear_val * (dist - spring_rest[offset + s])
                f = f_mag * direction
                ti.atomic_add(force[a][0], f[0])
                ti.atomic_add(force[a][1], f[1])
                ti.atomic_add(force[a][2], f[2])
                ti.atomic_add(force[b][0], -f[0])
                ti.atomic_add(force[b][1], -f[1])
                ti.atomic_add(force[b][2], -f[2])

    # 弯曲弹簧
    if enable_bend_field[None] == 1:
        offset = num_struct_springs[None] + num_shear_springs[None]
        for s in range(num_bend_springs[None]):
            a = spring_a[offset + s]
            b = spring_b[offset + s]
            delta = positions[b] - positions[a]
            dist = delta.norm()
            if dist > 1e-8:
                direction = delta / dist
                f_mag = ks_bend_val * (dist - spring_rest[offset + s])
                f = f_mag * direction
                ti.atomic_add(force[a][0], f[0])
                ti.atomic_add(force[a][1], f[1])
                ti.atomic_add(force[a][2], f[2])
                ti.atomic_add(force[b][0], -f[0])
                ti.atomic_add(force[b][1], -f[1])
                ti.atomic_add(force[b][2], -f[2])


# ============================================================
# ti.func: 处理球体碰撞
# ============================================================
@ti.func
def handle_sphere_collision(p, v):
    to_center = p - sphere_center
    dist = to_center.norm()
    p_new = p
    v_new = v
    if dist < sphere_radius and dist > 1e-8:
        normal = to_center / dist
        p_new = sphere_center + normal * sphere_radius
        v_normal = v.dot(normal)
        if v_normal < 0.0:
            v_new = v - v_normal * normal * 1.2
        else:
            v_new = v
    return p_new, v_new


# ============================================================
# Kernel 5: 显式欧拉积分 (Explicit Euler)
# ============================================================
@ti.kernel
def step_explicit():
    # 步骤1: 计算受力
    compute_spring_forces(pos, ks_struct, ks_shear, ks_bend)

    # 步骤2: 更新每个质点的位置和速度
    for idx in range(num_particles):
        if pinned[idx] == 1:
            continue

        # 累加重力和阻尼
        v = vel[idx]
        f = force[idx] + tm.vec3(0.0, -mass * 9.8, 0.0) - kd_field[None] * v

        # 加速度
        a = f / mass

        # 显式欧拉: v_{t+1} = v_t + a_t * dt
        #          x_{t+1} = x_t + v_t * dt
        v_new = v + a * dt
        x_new = pos[idx] + v * dt

        # 速度钳制
        v_new = clamp_velocity(v_new)

        # 球体碰撞
        if enable_collision_field[None] == 1:
            x_new, v_new = handle_sphere_collision(x_new, v_new)

        # 边界约束（不让布料跑太远）
        x_new.y = ti.max(x_new.y, -1.5)

        pos[idx] = x_new
        vel[idx] = v_new


# ============================================================
# Kernel 6: 半隐式欧拉积分 (Semi-Implicit / Symplectic Euler)
# ============================================================
@ti.kernel
def step_semi_implicit():
    # 步骤1: 计算受力
    compute_spring_forces(pos, ks_struct, ks_shear, ks_bend)

    # 步骤2: 更新每个质点的位置和速度
    for idx in range(num_particles):
        if pinned[idx] == 1:
            continue

        v = vel[idx]
        f = force[idx] + tm.vec3(0.0, -mass * 9.8, 0.0) - kd_field[None] * v

        a = f / mass

        # 半隐式欧拉: v_{t+1} = v_t + a_t * dt
        #            x_{t+1} = x_t + v_{t+1} * dt
        v_new = v + a * dt
        v_new = clamp_velocity(v_new)
        x_new = pos[idx] + v_new * dt

        # 球体碰撞
        if enable_collision_field[None] == 1:
            x_new, v_new = handle_sphere_collision(x_new, v_new)

        x_new.y = ti.max(x_new.y, -1.5)

        pos[idx] = x_new
        vel[idx] = v_new


# ============================================================
# Kernel 7: 隐式欧拉积分 (Implicit / Backward Euler)
# 使用定点迭代法近似求解
# ============================================================
@ti.kernel
def step_implicit():
    # 保存当前状态
    for idx in range(num_particles):
        pos_prev[idx] = pos[idx]
        vel_prev[idx] = vel[idx]

    # 定点迭代
    for _ in range(implicit_iters):
        # 使用当前估计的位置计算弹簧力
        compute_spring_forces(pos, ks_struct, ks_shear, ks_bend)

        for idx in range(num_particles):
            if pinned[idx] == 1:
                continue

            v = vel_prev[idx]
            f = force[idx] + tm.vec3(0.0, -mass * 9.8, 0.0) - kd_field[None] * vel[idx]

            a = f / mass

            # 隐式欧拉: v_{t+1} = v_t + a_{t+1} * dt
            #          x_{t+1} = x_t + v_{t+1} * dt
            v_new = v + a * dt
            v_new = clamp_velocity(v_new)
            x_new = pos_prev[idx] + v_new * dt

            # 球体碰撞
            if enable_collision_field[None] == 1:
                x_new, v_new = handle_sphere_collision(x_new, v_new)

            x_new.y = ti.max(x_new.y, -1.5)

            pos[idx] = x_new
            vel[idx] = v_new


# ============================================================
# Kernel 8: 更新渲染顶点
# ============================================================
@ti.kernel
def update_render_vertices():
    for idx in range(num_particles):
        cloth_vertices[idx] = pos[idx]


# ============================================================
# Kernel 9: 更新弹簧边渲染
# ============================================================
@ti.kernel
def update_edge_vertices():
    edge_count = 0
    # 渲染结构弹簧边
    for s in range(num_struct_springs[None]):
        if edge_count < MAX_EDGES:
            a = spring_a[s]
            b = spring_b[s]
            edge_vertices[edge_count * 2] = pos[a]
            edge_vertices[edge_count * 2 + 1] = pos[b]
            edge_count += 1
    # 渲染剪切弹簧边
    if enable_shear_field[None] == 1:
        offset = num_struct_springs[None]
        for s in range(num_shear_springs[None]):
            if edge_count < MAX_EDGES:
                a = spring_a[offset + s]
                b = spring_b[offset + s]
                edge_vertices[edge_count * 2] = pos[a]
                edge_vertices[edge_count * 2 + 1] = pos[b]
                edge_count += 1
    # 渲染弯曲弹簧边
    if enable_bend_field[None] == 1:
        offset = num_struct_springs[None] + num_shear_springs[None]
        for s in range(num_bend_springs[None]):
            if edge_count < MAX_EDGES:
                a = spring_a[offset + s]
                b = spring_b[offset + s]
                edge_vertices[edge_count * 2] = pos[a]
                edge_vertices[edge_count * 2 + 1] = pos[b]
                edge_count += 1
    num_edges_to_render[None] = edge_count


# ============================================================
# Kernel 10: 重置布料
# ============================================================
@ti.kernel
def reset_cloth():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x = (j - (N - 1) / 2.0) * grid_size
        y = 0.0
        z = i * grid_size
        pos[idx] = tm.vec3(x, y, z)
        vel[idx] = tm.vec3(0.0)
        force[idx] = tm.vec3(0.0)


# ============================================================
# 主程序
# ============================================================
def main():
    # 初始化场景
    init_positions()
    init_springs()
    init_render_indices()
    init_sphere_particles()

    integrator_type[None] = 1  # 默认半隐式欧拉
    paused[None] = 0
    reset_flag[None] = 0

    # 初始化 GUI 可控参数（Taichi field）
    kd_field[None] = 5.0
    enable_collision_field[None] = 0
    enable_shear_field[None] = 1
    enable_bend_field[None] = 1

    print(f"布料网格: {N}x{N} = {num_particles} 个质点")
    print(f"结构弹簧: {num_struct_springs[None]}")
    print(f"剪切弹簧: {num_shear_springs[None]}")
    print(f"弯曲弹簧: {num_bend_springs[None]}")
    print(f"总弹簧数: {num_struct_springs[None] + num_shear_springs[None] + num_bend_springs[None]}")
    print()

    integrator_names = ["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler"]
    print("控制说明:")
    print("  鼠标左键拖拽: 旋转视角")
    print("  鼠标滚轮: 缩放")
    print("  GUI 按钮: 切换积分器 / 暂停 / 重置")
    print("  当前默认积分器:", integrator_names[integrator_type[None]])

    # 创建窗口
    window = ti.ui.Window("Mass-Spring Cloth Simulation", (1280, 720), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # 设置相机初始位置
    camera.position(0.8, 0.5, 1.5)
    camera.lookat(0.0, -0.3, 0.5)
    camera.up(0.0, 1.0, 0.0)

    # 设置灯光
    scene.point_light(pos=(0.0, 5.0, 3.0), color=(0.8, 0.8, 0.8))
    scene.ambient_light((0.5, 0.5, 0.5))

    # GUI 参数存储（Python 侧）
    frame_count = 0
    while window.running:
        # 处理重置
        if reset_flag[None] == 1:
            reset_cloth()
            reset_flag[None] = 0

        # GUI Control Panel
        with window.GUI.sub_window("Integrator", 0.02, 0.02, 0.2, 0.14):
            window.GUI.text("Integrator Select")
            if window.GUI.button("1. Explicit Euler"):
                integrator_type[None] = 0
            if window.GUI.button("2. Semi-Implicit"):
                integrator_type[None] = 1
            if window.GUI.button("3. Implicit Euler"):
                integrator_type[None] = 2

        with window.GUI.sub_window("Control", 0.24, 0.02, 0.2, 0.12):
            if window.GUI.button("Pause / Resume"):
                paused[None] = 1 - paused[None]
            if window.GUI.button("Reset Cloth"):
                reset_flag[None] = 1
            status = "Paused" if paused[None] == 1 else "Running"
            window.GUI.text(f"Status: {status}")
            window.GUI.text(f"Frame: {frame_count}")

        with window.GUI.sub_window("Params", 0.02, 0.18, 0.2, 0.26):
            window.GUI.text(f"Integrator: {integrator_names[integrator_type[None]]}")
            window.GUI.text(f"Damping kd: {kd_field[None]:.1f}")
            if window.GUI.button("kd=1.0"):
                kd_field[None] = 1.0
            if window.GUI.button("kd=5.0"):
                kd_field[None] = 5.0
            if window.GUI.button("kd=10.0"):
                kd_field[None] = 10.0

        with window.GUI.sub_window("Extras", 0.02, 0.46, 0.2, 0.18):
            if window.GUI.button("Shear: " + ("ON" if enable_shear_field[None] == 1 else "OFF")):
                enable_shear_field[None] = 1 - enable_shear_field[None]
            if window.GUI.button("Bending: " + ("ON" if enable_bend_field[None] == 1 else "OFF")):
                enable_bend_field[None] = 1 - enable_bend_field[None]
            if window.GUI.button("Collision: " + ("ON" if enable_collision_field[None] == 1 else "OFF")):
                enable_collision_field[None] = 1 - enable_collision_field[None]

        # 物理更新
        if paused[None] == 0:
            # 子步循环（每帧多次物理更新）
            num_substeps = 5
            for _ in range(num_substeps):
                if integrator_type[None] == 0:
                    step_explicit()
                elif integrator_type[None] == 1:
                    step_semi_implicit()
                elif integrator_type[None] == 2:
                    step_implicit()
            frame_count += 1

        # 更新渲染数据
        update_render_vertices()
        update_edge_vertices()

        # 设置场景
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # 渲染质点（小球）
        scene.particles(pos, radius=0.005, color=(0.2, 0.4, 1.0))

        # 渲染弹簧（线）
        if num_edges_to_render[None] > 0:
            scene.lines(edge_vertices, width=1.5,
                       color=(0.8, 0.8, 0.8),
                       vertex_count=num_edges_to_render[None] * 2)

        # 渲染碰撞球体（用更小更暗的粒子）
        if enable_collision_field[None] == 1:
            scene.particles(sphere_particles, radius=0.005,
                           color=(0.6, 0.4, 0.2))

        # 渲染到画布
        canvas.scene(scene)

        window.show()


if __name__ == "__main__":
    main()