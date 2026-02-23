"""HTP50cork — GUI 실행/스레딩 흐름 발췌(요약)

- GUI 프리징을 방지하기 위해 계산(solve)을 워커 스레드에서 수행하고,
  메인 스레드는 after() 기반 폴링으로 진행률을 업데이트한다.
- 전체 원본은 code/app.py를 참고.
"""

import threading
import time


class HeatTransferApp:
    def run_simulation(self):
        # 1) 입력값 검증 및 파라미터 구성
        params = {
            'L1': float(self.entries['L1'].get()),
            'L2': float(self.entries['L2'].get()),
            'N1': int(self.entries['N1'].get()),
            'N2': int(self.entries['N2'].get()),
            't_final': float(self.entries['t_final'].get()),
            'T_init': float(self.entries['T_init'].get()),
            'T_simple': float(self.entries['T_simple'].get()),
            'pyro_mode': self.pyro_mode_var.get(),
        }

        # 2) 열분해 포함/미포함 솔버 2개 구성
        solver_pyro    = HeatSolver(params, mat_data_pyro,    self.bc_data, enable_pyrolysis=True)
        solver_no_pyro = HeatSolver(params, mat_data_no_pyro, self.bc_data, enable_pyrolysis=False)

        # 3) 워커 스레드에서 계산 실행
        thread = threading.Thread(target=self._run_solver_thread,
                                  args=(solver_pyro, solver_no_pyro))
        thread.daemon = True
        thread.start()

        # 4) 메인 스레드: 진행률 폴링
        self.monitor_progress(solver_pyro, thread)

    def monitor_progress(self, solver, thread):
        if thread.is_alive():
            t_curr = getattr(solver, 't_current', 0.0)
            t_final = solver.p['t_final']
            self.progress_bar.set(min(t_curr / t_final, 1.0))
            self.after(100, lambda: self.monitor_progress(solver, thread))
        else:
            self.progress_bar.set(1.0)

    def _run_solver_thread(self, solver1, solver2):
        start_t = time.time()
        sol1 = solver1.solve()
        sol2 = solver2.solve()
        elapsed = time.time() - start_t
        self.after(0, lambda: self._on_simulation_complete(solver1, sol1, sol2, elapsed))
