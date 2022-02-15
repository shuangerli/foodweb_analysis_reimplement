import numpy as np
import csv
import sys


#Avoid log0=-inf
def c_log(x):
	return np.log10(x + 0.0000001)


def read_network(filepath):
	nt = []
	with open(filepath) as csvfile:
		csvreader = csv.reader(csvfile, delimiter = ",")

		for row in csvreader:
			row_int = [int(s) for s in row]
			nt.append(row_int)

	nt = np.array(nt, dtype = int)

	return nt


#Initialize chromosome population
def initialize_pop(N, S, k):
	pop = np.random.randint(low = 0, high = k, size = (N, S))
	return pop

#Calculate AIC for a particular grouping (ch)
#p_ij = l_ij/S_iS_j
def calc_AIC(nt, ch, k, S):
	k_dict = {i:[] for i in range(k)}
	for i in range(S):
		k_dict[ch[i]].append(i)

	loglikehood = 0.
	for i in range(k):
		for j in range(k):
			nodes_i, nodes_j = k_dict[i], k_dict[j]
			s_i, s_j = len(nodes_i), len(nodes_j)
			l_ij = sum([nt[x, y] for x in nodes_i for y in nodes_j])
			p_ij = l_ij / (s_i * s_j + 0.0000001)
			loglikehood += (l_ij * c_log(p_ij) + (s_i * s_j - l_ij) * (c_log(1. - p_ij)))

	AIC = 2. * k * k + 2. * S - (2. * loglikehood)

	return AIC


def locally_optimize(pop, nt, k, S, N):
	#Iterate to locally optimize each chromosome
	for i in range(N):
		ch = pop[i]
		best_score = calc_AIC(nt, ch, k, S)
		best_neighbor = ch.copy()
		cont = True
		cycles = 0

		#Hill climbing: search for the best neighbor
		while cont:
			cycles += 1
			starter_best_score = best_score
			starter_best_neighbor = best_neighbor
			for position in range(S):
				for group in range(k):
					#try every possible neighbor differing by this position
					if group != ch[position]:
						temp_ch = starter_best_neighbor.copy()
						temp_ch[position] = group
						temp_score = calc_AIC(nt, temp_ch, k, S)
						if temp_score < best_score:
							best_score = temp_score
							best_neighbor = temp_ch.copy()
			if starter_best_score == best_score or cycles > 1000:
				cont = False

		pop[i] = best_neighbor.copy()

	return pop


def simulate_pop_one_gen(pop, nt, N, S, M, k):
	#Hill climbing
	pop = locally_optimize(pop, nt, k, S, N)

	#Reproduction
	AICs = np.array([calc_AIC(nt, ch, k, S) for ch in pop])
	
	fitnesses = 1. / AICs
	wbar = np.mean(fitnesses)
	freqs = (1. / N) * (fitnesses / wbar)
	new_nums = np.random.multinomial(N, pvals = freqs)

	#Mutation
	new_pop = np.array([pop[i] for i in range(N) for j in range(new_nums[i])])
	mutations = np.random.binomial(n = 1, p = M, size = (N, S))
	mutations_index = np.where(mutations == 1)
	targets = np.random.randint(low = 0, high = k, size = len(mutations_index[0]))

	for i in range(len(targets)):
		new_pop[mutations_index[0][i], mutations_index[1][i]] = targets[i]

	return new_pop


def simulate_pop(pop, nt, N, S, M, k, MAX_GEN):
	cont = True
	gens = 0

	while cont:
		gens += 1
		pop = simulate_pop_one_gen(pop, nt, N, S, M, k)
		print(gens, flush = True)
		neighbor_equal = [np.array_equal(pop[i], pop[i + 1]) for i in range(N - 1)]
		if all(neighbor_equal) or gens > MAX_GEN:
			cont = False
			print(gens, flush = True)

	score = calc_AIC(nt, pop[0], k, S)
	grouping = pop[0]

	return score, grouping




def main():
	SEED = 1024
	MAX_GEN = 1000
	N = 1000 		#size of the chromosome's population
	M = 3. 	        #expected number of mutations in each generation
	MIN_K = 1       #minimum number of groups
	MAX_K = 6       #maximum number of groups
	S = 0 			#initialize num of nodes in the network

	if len(sys.argv) > 1:
		NETWORK_PATH = sys.argv[1]
	else: 
		NETWORK_PATH = "tests/adj_m_small.csv"
	BEST_SCORES = np.ones(MAX_K) * np.inf
	BEST_GROUPING = []


	######################SCRATCH TESTS######################
	TEST = True
	if TEST:
		N = 10
		MIN_K = 11
		MAX_K = 15
		MAX_GEN = 100
		M = 3.

	#########################################################

	np.random.seed(SEED)

	#Read network
	nt = read_network(NETWORK_PATH) #adj matrix; np arr
	S = len(nt)
	MAX_K = min(MAX_K, S) #cannot have more groups than nodes
	assert(S > 0)
	M = M / (N * S)


	#Grouping
	for k in range(MIN_K, MAX_K + 1):
		pop = initialize_pop(N, S, k)
		best_score, best_group = simulate_pop(pop, nt, N, S, M, k, MAX_GEN)
		print("k =", k, "AIC =", best_score, "grouping:", best_group, flush = True)
		





	

if __name__ == "__main__":
	main()



