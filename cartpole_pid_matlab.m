% =============================================================================
% cartpole_pid_matlab.m
% CartPole PID Controller — MATLAB実装（Isaac Sim 比較用）
%
% 使用方法:
%   >> cartpole_pid_matlab          % 実行 + CSV出力 + プロット
%   >> cartpole_pid_matlab(true)    % CSV のみ（プロットなし）
%
% 出力:
%   ./output/matlab_pid_log.csv
%   ./output/matlab_pid_plot.png   (プロット有効時)
%
% ─────────────────────────────────────────────────────────────────────────────
% 物理モデル（Isaac Sim cartpole.usd に合わせた値）
%   M = 1.0 kg  : カート質量
%   m = 0.1 kg  : ポール質量
%   l = 0.5 m   : ポール重心までの長さ (全長 1.0m の半分)
%   g = 9.81 m/s²
%   dt= 1/120 s : Isaac Sim physics_dt と統一
%
% 運動方程式（完全非線形, Florian 2007）:
%   I_pole = (4/3)*m*l²
%   D      = (M+m)*I_pole - (m*l*cosθ)²
%   ẍ  = [I_pole*(F + m*l*θ̇²sinθ) - m*l*cosθ*m*g*l*sinθ] / D
%   θ̈  = [(M+m)*m*g*l*sinθ - m*l*cosθ*(F + m*l*θ̇²sinθ)] / D
%
% PID 制御則:
%   F = Kp_pole*θ + Ki_pole*∫θdt + Kd_pole*θ̇   (ポール安定化)
%     + Kp_cart*x + Kd_cart*ẋ                    (カート位置回復)
%
% PIDゲイン導出: 状態空間 Q=diag([1,0.1,100,10]) R=0.001 でLQR設計し
%               K[cart_x, cart_v, pole, omega] = [-31.6, -56.9, -497.6, -141.4]
%               を参考にPIDゲインに変換
% =============================================================================

function cartpole_pid_matlab(csv_only)

    if nargin < 1, csv_only = false; end

    % =========================================================================
    % 1. パラメータ定義
    % =========================================================================
    % 物理パラメータ
    M  = 1.0;     % カート質量 [kg]
    m  = 0.1;     % ポール質量 [kg]
    l  = 0.5;     % ポール重心までの長さ [m]
    g  = 9.81;    % 重力加速度 [m/s²]
    dt = 1/120;   % タイムステップ [s] ← Isaac Sim physics_dt と統一

    % シミュレーション設定
    T_sim          = 60.0;
    N              = round(T_sim / dt);
    init_angle_deg = 5.0;    % ポール初期角度 [deg]

    % PIDゲイン（LQR由来: Q=diag([1,0.1,100,10]) R=0.001）
    Kp_pole =  497.62;   % ポール比例ゲイン
    Ki_pole =    5.00;   % ポール積分ゲイン
    Kd_pole =  141.36;   % ポール微分ゲイン
    Kp_cart =   31.62;   % カート位置比例ゲイン
    Kd_cart =   56.91;   % カート速度微分ゲイン
    integral_limit = 0.5;
    force_limit    = 500.0;   % [N]

    % =========================================================================
    % 2. 完全非線形 運動方程式（内部関数）
    % =========================================================================
    function dxdt = ode_fn(~, xs, F)
        v_   = xs(2);  th_  = xs(3);  w_   = xs(4);
        st   = sin(th_);  ct  = cos(th_);
        Ip   = (4/3)*m*l^2;
        D    = (M+m)*Ip - (m*l*ct)^2;
        xdd  = (Ip*(F + m*l*w_^2*st) - m*l*ct*m*g*l*st) / D;
        thdd = ((M+m)*m*g*l*st - m*l*ct*(F + m*l*w_^2*st)) / D;
        dxdt = [v_; xdd; w_; thdd];
    end

    % RK4 1ステップ
    function xs_next = rk4_step(t_, xs, F)
        k1 = ode_fn(t_,       xs,           F);
        k2 = ode_fn(t_+dt/2,  xs+dt/2*k1,  F);
        k3 = ode_fn(t_+dt/2,  xs+dt/2*k2,  F);
        k4 = ode_fn(t_+dt,    xs+dt*k3,    F);
        xs_next = xs + (dt/6)*(k1+2*k2+2*k3+k4);
    end

    % =========================================================================
    % 3. ログ用配列（事前確保）
    % =========================================================================
    log_time  = zeros(N,1); log_cx = zeros(N,1); log_cv = zeros(N,1);
    log_prad  = zeros(N,1); log_pdeg = zeros(N,1); log_pw = zeros(N,1);
    log_F     = zeros(N,1); log_up = zeros(N,1); log_ui = zeros(N,1);
    log_ud    = zeros(N,1);

    % =========================================================================
    % 4. シミュレーションループ
    % =========================================================================
    xs       = [0.0; 0.0; deg2rad(init_angle_deg); 0.0];
    intg     = 0.0;
    N_actual = N;

    for k = 1:N
        t_k  = (k-1)*dt;
        cx   = xs(1);  cv  = xs(2);  prad = xs(3);  pw = xs(4);

        % 離散 PID
        intg = max(-integral_limit, min(integral_limit, intg + prad*dt));
        up   = Kp_pole * prad;
        ui   = Ki_pole * intg;
        ud   = Kd_pole * pw;
        F    = max(-force_limit, min(force_limit, up+ui+ud + Kp_cart*cx+Kd_cart*cv));

        % ログ
        log_time(k)=t_k; log_cx(k)=cx; log_cv(k)=cv;
        log_prad(k)=prad; log_pdeg(k)=rad2deg(prad); log_pw(k)=pw;
        log_F(k)=F; log_up(k)=up; log_ui(k)=ui; log_ud(k)=ud;

        % RK4
        xs = rk4_step(t_k, xs, F);

        % 終了判定
        if abs(xs(3)) > pi/2 || abs(xs(1)) > 3.0
            fprintf('[WARN] 制御失敗 @ t=%.3f s (θ=%.1f°, x=%.3f m)\n', ...
                    t_k, rad2deg(xs(3)), xs(1));
            N_actual = k;  break
        end
    end

    % ログ切り詰め
    idx = 1:N_actual;
    log_time=log_time(idx); log_cx=log_cx(idx); log_cv=log_cv(idx);
    log_prad=log_prad(idx); log_pdeg=log_pdeg(idx); log_pw=log_pw(idx);
    log_F=log_F(idx); log_up=log_up(idx); log_ui=log_ui(idx); log_ud=log_ud(idx);

    fprintf('[INFO] 完了: %d steps / %.2f s\n', N_actual, log_time(end));

    % =========================================================================
    % 5. CSV 出力（列名は Isaac Sim Standalone スクリプトと完全統一）
    % =========================================================================
    out_dir  = './output';
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    csv_path = fullfile(out_dir, 'matlab_pid_log.csv');

    fid = fopen(csv_path, 'w');
    fprintf(fid, 'time_s,cart_x_m,cart_v_ms,pole_rad,pole_deg,pole_omega_rads,force_N,pid_up,pid_ui,pid_ud\n');
    fmt = '%.6f,%.6f,%.6f,%.6f,%.4f,%.6f,%.4f,%.4f,%.4f,%.4f\n';
    for k = 1:N_actual
        fprintf(fid, fmt, log_time(k), log_cx(k), log_cv(k), ...
                log_prad(k), log_pdeg(k), log_pw(k), log_F(k), ...
                log_up(k), log_ui(k), log_ud(k));
    end
    fclose(fid);
    fprintf('[CSV] 保存完了: %s\n', csv_path);

    if csv_only, return; end

    % =========================================================================
    % 6. プロット
    % =========================================================================
    fig = figure('Name','CartPole PID — MATLAB','NumberTitle','off', ...
                 'Position',[100 80 1050 700]);

    subplot(3,2,1);
    plot(log_time, log_pdeg, 'b-', 'LineWidth',1.3);
    yline(0,'k--'); xlabel('Time [s]'); ylabel('deg');
    title('Pole Angle'); grid on;

    subplot(3,2,2);
    plot(log_time, log_cx, 'r-', 'LineWidth',1.3);
    yline(0,'k--'); xlabel('Time [s]'); ylabel('m');
    title('Cart Position'); grid on;

    subplot(3,2,3);
    plot(log_time, log_pw, 'm-', 'LineWidth',1.0);
    xlabel('Time [s]'); ylabel('rad/s');
    title('Pole Angular Velocity'); grid on;

    subplot(3,2,4);
    plot(log_time, log_F, 'k-', 'LineWidth',1.0);
    xlabel('Time [s]'); ylabel('N');
    title('Control Force'); grid on;

    subplot(3,2,5);
    hold on;
    plot(log_time, log_up, 'b-',  'LineWidth',1.0, 'DisplayName','P');
    plot(log_time, log_ui, 'g--', 'LineWidth',1.0, 'DisplayName','I');
    plot(log_time, log_ud, 'r:',  'LineWidth',1.2, 'DisplayName','D');
    hold off; legend('Location','northeast');
    xlabel('Time [s]'); ylabel('N'); title('PID Breakdown'); grid on;

    subplot(3,2,6);
    plot(log_pdeg, log_pw, 'b-', 'LineWidth',0.8); hold on;
    scatter(log_pdeg(1),   log_pw(1),   60,'g','filled','DisplayName','Start');
    scatter(log_pdeg(end), log_pw(end), 60,'r','filled','DisplayName','End');
    hold off; legend('Location','best');
    xlabel('Pole angle [deg]'); ylabel('Pole omega [rad/s]');
    title('Phase Portrait'); grid on;

    sgtitle(sprintf('CartPole PID  Kp=%.1f Ki=%.1f Kd=%.1f | M=%.1f m=%.2f l=%.1f | init=%.1f°', ...
            Kp_pole,Ki_pole,Kd_pole,M,m,l,init_angle_deg), ...
            'FontSize',10,'FontWeight','bold');

    png_path = fullfile(out_dir, 'matlab_pid_plot.png');
    exportgraphics(fig, png_path, 'Resolution', 150);
    fprintf('[PLOT] 保存: %s\n', png_path);

end
