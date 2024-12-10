function [f_opt, x_opt, y_opt, z_opt] = admm_solver(m, K, lambda, mu, t, max_iter, tol)
    % 输入:
    % m - 输入数据 (N维向量)
    % K - 系统矩阵
    % lambda - 正则化参数
    % mu - ADMM 参数
    % t - 指示变量 (N维向量, 取值为0或1)
    % max_iter - 最大迭代次数
    % tol - 收敛阈值
    
    % 初始化
    N = length(m); % 数据长度
    f = zeros(N, 1); % 初始化 f_i
    x = zeros(N, 1); % 初始化 x_i
    y = zeros(N, 1); % 初始化 y_i
    z = zeros(N, 1); % 初始化 z_i
    g_x = zeros(N, 1); % 拉格朗日乘子 g_x
    g_y = zeros(N, 1); % 拉格朗日乘子 g_y
    g_z = zeros(N, 1); % 拉格朗日乘子 g_z
    
    % 预计算
    KTK = K' * K;
    inv_matrix = (KTK + mu * eye(N)) \ eye(N); % 预计算逆矩阵，用于 x_i 更新

    % ADMM 主循环
    for iter = 1:max_iter
        % Step 1: 更新 f_i
        for i = 1:N
            if t(i) == 1
                % 如果 t_i = 1，f_i 受制于 x_i
                f(i) = (x(i) + g_x(i) + y(i) + g_y(i) + z(i) + g_z(i)) / 3;
            else
                % 如果 t_i = 0，仅考虑 y_i 和 z_i
                f(i) = (y(i) + g_y(i) + z(i) + g_z(i)) / 2;
            end
        end
        
        % Step 2: 更新 x_i
        for i = 1:N
            x(i) = inv_matrix * (K' * m + mu * (t(i) * f(i) - g_x(i)));
        end
        
        % Step 3: 更新 y_i
        y = max(f - g_y, 0); % 非负约束
        
        % Step 4: 更新 z_i
        z = update_z(f, g_z, z, mu, lambda, N);
        
        % Step 5: 更新拉格朗日乘子
        g_x = g_x + t .* f - x; % 根据 t 调整约束
        g_y = g_y + f - y;
        g_z = g_z + f - z;
        
        % Step 6: 检查收敛条件
        primal_residual = norm(t .* f - x) + norm(f - y) + norm(f - z);
        dual_residual = mu * (norm(f - x) + norm(y - z));
        if primal_residual < tol && dual_residual < tol
            break;
        end
    end

    % 输出结果
    f_opt = f;
    x_opt = x;
    y_opt = y;
    z_opt = z;
end
