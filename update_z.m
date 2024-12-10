function z_new = update_z(f, g_z, z_old, mu, lambda, N)
    % 更新 z_i 的辅助函数
    % 输入:
    % f - 当前 f 向量
    % g_z - 当前的拉格朗日乘子 g_z
    % z_old - 上一次迭代的 z
    % mu, lambda - ADMM 参数
    % N - 点的总数

    z_new = zeros(N, 1); % 初始化 z_new

    % 逐点更新 z_i
    for i = 1:N
        % 计算邻域总和，假设邻域为简单的相邻点
        neighbor_sum = 0;
        num_neighbors = 0;
        if i > 1
            neighbor_sum = neighbor_sum + z_old(i - 1);
            num_neighbors = num_neighbors + 1;
        end
        if i < N
            neighbor_sum = neighbor_sum + z_old(i + 1);
            num_neighbors = num_neighbors + 1;
        end

        % 更新 z_i
        z_new(i) = (mu * (f(i) - g_z(i)) + lambda * neighbor_sum) / (mu + lambda * num_neighbors);
    end
end