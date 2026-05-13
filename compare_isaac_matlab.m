% =============================================================================
% compare_isaac_matlab.m
% Isaac Sim CSV と MATLAB CSV を読み込んで比較プロット + RMSE 統計を出力
%
% 前提:
%   ./output/isaac_pid_log.csv   — Isaac Sim Standalone の出力
%   ./output/matlab_pid_log.csv  — cartpole_pid_matlab.m の出力
%
% 使用方法:
%   >> compare_isaac_matlab
%
% 出力:
%   ./output/comparison_plot.png
%   コンソールに RMSE サマリーを表示
%
% =============================================================================

function compare_isaac_matlab()

    isaac_csv  = './output/isaac_pid_log.csv';
    matlab_csv = './output/matlab_pid_log.csv';

    if ~isfile(isaac_csv)
        error('Isaac Sim CSV が見つかりません: %s\n→ Standalone スクリプトを実行してください。', isaac_csv);
    end
    if ~isfile(matlab_csv)
        error('MATLAB CSV が見つかりません。\n→ cartpole_pid_matlab を先に実行してください。');
    end

    IA = readtable(isaac_csv,  'VariableNamingRule', 'preserve');
    MA = readtable(matlab_csv, 'VariableNamingRule', 'preserve');

    fprintf('[INFO] Isaac Sim : %d rows (%.2f s)\n', height(IA), IA.time_s(end));
    fprintf('[INFO] MATLAB    : %d rows (%.2f s)\n', height(MA), MA.time_s(end));

    % ─ 色定義 ─
    c_i = [0.00, 0.45, 0.74];   % Isaac Sim = 青
    c_m = [0.85, 0.33, 0.10];   % MATLAB    = 橙赤

    fig = figure('Name', 'Isaac Sim vs MATLAB — PID CartPole', ...
                 'NumberTitle', 'off', 'Position', [40 40 1150 780]);

    % ─────────────────────────────────────────────────────────────
    % subplot 1: Pole Angle
    % ─────────────────────────────────────────────────────────────
    subplot(3, 2, 1);
    hold on;
    plot(IA.time_s, IA.pole_deg,  '-',  'Color', c_i, 'LineWidth', 1.6, ...
         'DisplayName', 'Isaac Sim');
    plot(MA.time_s, MA.pole_deg,  '--', 'Color', c_m, 'LineWidth', 1.6, ...
         'DisplayName', 'MATLAB');
    yline(0, 'k:', 'LineWidth', 0.7);
    hold off; legend('Location', 'northeast');
    xlabel('Time [s]'); ylabel('deg');
    title('Pole Angle'); grid on;

    % ─────────────────────────────────────────────────────────────
    % subplot 2: Cart Position
    % ─────────────────────────────────────────────────────────────
    subplot(3, 2, 2);
    hold on;
    plot(IA.time_s, IA.cart_x_m, '-',  'Color', c_i, 'LineWidth', 1.6, ...
         'DisplayName', 'Isaac Sim');
    plot(MA.time_s, MA.cart_x_m, '--', 'Color', c_m, 'LineWidth', 1.6, ...
         'DisplayName', 'MATLAB');
    yline(0, 'k:', 'LineWidth', 0.7);
    hold off; legend('Location', 'northeast');
    xlabel('Time [s]'); ylabel('m');
    title('Cart Position'); grid on;

    % ─────────────────────────────────────────────────────────────
    % subplot 3: Pole Angular Velocity
    % ─────────────────────────────────────────────────────────────
    subplot(3, 2, 3);
    hold on;
    plot(IA.time_s, IA.pole_omega_rads, '-',  'Color', c_i, 'LineWidth', 1.3, ...
         'DisplayName', 'Isaac Sim');
    plot(MA.time_s, MA.pole_omega_rads, '--', 'Color', c_m, 'LineWidth', 1.3, ...
         'DisplayName', 'MATLAB');
    hold off; legend('Location', 'northeast');
    xlabel('Time [s]'); ylabel('rad/s');
    title('Pole Angular Velocity'); grid on;

    % ─────────────────────────────────────────────────────────────
    % subplot 4: Control Force
    % ─────────────────────────────────────────────────────────────
    subplot(3, 2, 4);
    hold on;
    plot(IA.time_s, IA.force_N, '-',  'Color', c_i, 'LineWidth', 1.3, ...
         'DisplayName', 'Isaac Sim');
    plot(MA.time_s, MA.force_N, '--', 'Color', c_m, 'LineWidth', 1.3, ...
         'DisplayName', 'MATLAB');
    hold off; legend('Location', 'northeast');
    xlabel('Time [s]'); ylabel('N');
    title('Control Force'); grid on;

    % ─────────────────────────────────────────────────────────────
    % subplot 5: 差分 — Pole Angle (Isaac - MATLAB)
    % ─────────────────────────────────────────────────────────────
    subplot(3, 2, 5);
    N_com = min(height(IA), height(MA));
    t_com = IA.time_s(1:N_com);
    d_pole = IA.pole_deg(1:N_com) - MA.pole_deg(1:N_com);
    rmse_p = sqrt(mean(d_pole.^2));
    plot(t_com, d_pole, 'k-', 'LineWidth', 1.0);
    xlabel('Time [s]'); ylabel('deg');
    title('Pole Angle Difference  (Isaac − MATLAB)'); grid on;
    text(0.05, 0.88, sprintf('RMSE = %.5f  deg', rmse_p), ...
         'Units', 'normalized', 'FontSize', 9, 'FontWeight', 'bold');

    % ─────────────────────────────────────────────────────────────
    % subplot 6: 差分 — Cart Position (Isaac - MATLAB)
    % ─────────────────────────────────────────────────────────────
    subplot(3, 2, 6);
    d_cart = IA.cart_x_m(1:N_com) - MA.cart_x_m(1:N_com);
    rmse_c = sqrt(mean(d_cart.^2));
    plot(t_com, d_cart, 'k-', 'LineWidth', 1.0);
    xlabel('Time [s]'); ylabel('m');
    title('Cart Position Difference  (Isaac − MATLAB)'); grid on;
    text(0.05, 0.88, sprintf('RMSE = %.5f  m', rmse_c), ...
         'Units', 'normalized', 'FontSize', 9, 'FontWeight', 'bold');

    % ─ タイトル ─
    sgtitle('Isaac Sim 5.1.0  vs  MATLAB — CartPole PID Comparison', ...
            'FontSize', 12, 'FontWeight', 'bold');

    % ─ PNG 保存 ─
    out_dir  = './output';
    png_path = fullfile(out_dir, 'comparison_plot.png');
    exportgraphics(fig, png_path, 'Resolution', 150);
    fprintf('[PLOT] 保存: %s\n', png_path);

    % ─────────────────────────────────────────────────────────────
    % サマリー出力
    % ─────────────────────────────────────────────────────────────
    fprintf('\n========================================\n');
    fprintf('  比較サマリー\n');
    fprintf('  共通サンプル数 : %d  (%.2f s)\n', N_com, t_com(end));
    fprintf('  ポール角度 RMSE: %.5f deg\n', rmse_p);
    fprintf('  カート位置 RMSE: %.5f m\n',   rmse_c);
    fprintf('\n  差異の主な原因:\n');
    fprintf('  1. Isaac Sim の PhysX は離散積分器（implicit Euler 系）;\n');
    fprintf('     MATLAB RK4 と積分誤差が蓄積する。\n');
    fprintf('  2. cartpole.usd の実際の質量・慣性はジオメトリから\n');
    fprintf('     自動計算されるため、MATLAB の手動設定値と完全一致しない。\n');
    fprintf('  3. 両者が安定収束した後の差はゼロに近づく。\n');
    fprintf('========================================\n');

end
