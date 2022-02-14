import numpy as np



def main():
	SEED = 1024
	Ns = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11, 12]] 	#define nodes in each group
	N = 13
	dense_connects = [[0, 1], [0, 2], [1, 3]]				#define densely connected groups (directed)
	sparse_connects = [[1, 2]]						#define loosely connected groups (directed)
	p_dense = 0.9									#define prob edges a->b
	p_sparse = 0.2
	filename = "adj_m_med.csv"

	np.random.seed(SEED)

	adj_m = np.zeros((N, N), dtype = int)

	#Draw edges
	for i in range(len(dense_connects)):
		a, b = Ns[dense_connects[i][0]], Ns[dense_connects[i][1]]
		for u in a:
			for v in b:
				edge = np.random.binomial(n = 1, p = p_dense, size = 1)
				if edge:
					adj_m[u, v] = 1

	for i in range(len(sparse_connects)):
		a, b = Ns[sparse_connects[i][0]], Ns[sparse_connects[i][1]]
		for u in a:
			for v in b:
				edge = np.random.binomial(n = 1, p = p_sparse, size = 1)
				if edge:
					adj_m[u, v] = 1

	adj_m = adj_m.astype(int)
	np.savetxt(filename, adj_m, fmt='%i', delimiter=",")




if __name__ == "__main__":
	main()