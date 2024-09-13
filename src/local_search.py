class GenericHillClimbSolver:
    def __init__(self, memory_size=0, solution_pool_size=1):
        self.memory_size = memory_size
        self.solution_pool_size = solution_pool_size

    def init_memory_bank(self, memory_size):
        return []

    def init_solution(self, solution_size):
        pass

    def make_options(self, solutions):
        pass

    def evaluation_function(self, solution, distance_matrix, **kwargs):
        pass

    def select_best_options(self, options, distance_matrix):
        pass

    def solve(self, distance_matrix):
        best_solutions = self.init_solution(distance_matrix.shape[0])
        assert isinstance(best_solutions, list) and len(best_solutions) > 0

        record = self.evaluation_function(best_solutions[0], distance_matrix)
        assert record <= 0, "We use negative values for MAX optimization"

        while True:
            print(record)
            options = self.make_options(best_solutions)

            best_options = self.select_best_options(options, distance_matrix)

            session_record = self.evaluation_function(best_options[0], distance_matrix)

            if session_record <= record:
                return best_solutions[0], record
            best_solutions = best_options
            record = session_record
