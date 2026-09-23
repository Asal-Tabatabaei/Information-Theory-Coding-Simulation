import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys

# =====================================================================
# PHASE 1 & 2: Dynamic Source, Validation & Huffman Coding
# =====================================================================
def generate_dynamic_huffman(probs_dict):
    nodes = [[prob, [sym]] for sym, prob in probs_dict.items()]
    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x[0])
        left = nodes.pop(0)
        right = nodes.pop(0)
        
        for sym in left[1]:
            item_code[sym] = '0' + item_code.get(sym, '')
        for sym in right[1]:
            item_code[sym] = '1' + item_code.get(sym, '')
            
        combined_symbols = left[1] + right[1]
        nodes.append([left[0] + right[0], combined_symbols])
    return item_code

item_code = {}

def get_dynamic_huffman_dict(symbols, probabilities, L):
    ext_symbols = list(itertools.product(symbols, repeat=L))
    ext_probs = [np.prod(p) for p in itertools.product(probabilities, repeat=L)]
    probs_dict = dict(zip(ext_symbols, ext_probs))
    
    global item_code
    item_code = {sym: '' for sym in ext_symbols}
    generate_dynamic_huffman(probs_dict)
    return item_code

# PHASE 3: Dynamic Systematic Hamming Matrices [A | I_r]

def generate_hamming_matrices(r):
    n = 2**r - 1
    k = n - r
    
    H_base = np.zeros((r, n), dtype=int)
    for i in range(1, n + 1):
        H_base[:, i-1] = [int(x) for x in format(i, f'0{r}b')]
    
    id_cols = []
    for i in range(r):
        vec = np.zeros(r, dtype=int)
        vec[i] = 1
        for j in range(n):
            if np.array_equal(H_base[:, j], vec):
                id_cols.append(j)
                break
                
    remaining_cols = [x for x in range(n) if x not in id_cols]
    
    A = H_base[:, remaining_cols]
    I_r = H_base[:, id_cols]
    H = np.hstack((A, I_r))
    G = np.hstack((np.eye(k, dtype=int), A.T))
    
    return G, H, n, k

# =====================================================================
# PHASE 4: BSC Channel & Syndrome Decoding

def simulate_bsc(bits, p_err):
    noise = np.random.rand(len(bits)) < p_err
    return (bits + noise) % 2

def decode_hamming(received, H, n, k):
    corrected = []
    for i in range(0, len(received), n):
        block = received[i:i+n]
        if len(block) < n: break
        
        syndrome = np.dot(block, H.T) % 2
        
        if np.any(syndrome):
            for col in range(n):
                if np.array_equal(H[:, col], syndrome):
                    block[col] = (block[col] + 1) % 2
                    break
        corrected.extend(block[:k])
    return np.array(corrected)



def main():
    print("\n=======================================================")
    print("    INFORMATION THEORY & CODING SIMULATION SYSTEM    ")
    print("=======================================================")
    print("1. Standalone Test: Phase 2 (Dynamic Huffman Codes)")
    print("2. Standalone Test: Phase 3 (Dynamic Hamming Matrices)")
    print("3. Standalone Test: Phase 4 (BSC Channel Noise)")
    print("4. Run Full Chain Simulation & Plot Performance Chart")
    print("=======================================================")
    
    choice = input("Please select an option (1-4): ")
    
    if choice == '1':
        print("\n--- Phase 2: Dynamic Huffman Configuration ---")
        try:
            q = int(input("Enter number of source symbols (q): "))
            if q <= 1:
                print(" Validation Error: Number of symbols (q) must be greater than 1!")
                sys.exit()
            
            symbols = [f"S{i+1}" for i in range(q)]
            probabilities = []
            for i in range(q):
                p_val = float(input(f"  Enter probability for symbol {symbols[i]}: "))
                if p_val < 0 or p_val > 1:
                    print(" Validation Error: Individual probabilities must be between 0 and 1!")
                    sys.exit()
                probabilities.append(p_val)
                
            if not np.isclose(sum(probabilities), 1.0, atol=1e-5):
                print(f" Validation Error: Sum of probabilities must equal 1.0! (Current sum: {sum(probabilities)})")
                sys.exit()
                
            L = int(input("Enter extension order L (1 or 2): "))
            if L <= 0:
                print(" Validation Error: Extension order L must be a positive integer!")
                sys.exit()
                
            codebook = get_dynamic_huffman_dict(symbols, probabilities, L)
            print(f"\nGenerated {len(codebook)} codes for extension L={L}:")
            for k, v in list(codebook.items())[:8]:
                print(f"  Block {k} -> Binary Code: {v}")
        except ValueError:
            print(" Validation Error: Invalid numeric input format typed!")
            sys.exit()
            
    elif choice == '2':
        print("\n--- Phase 3: Dynamic Hamming Matrix Generation ---")
        try:
            r = int(input("Enter number of parity bits r (e.g., 3 or 4): "))
            if r < 2:
                print(" Validation Error: Parity bits (r) must be at least 2!")
                sys.exit()
            G, H, n, k = generate_hamming_matrices(r)
            print(f"\n Structure: Hamming({n}, {k}) | Code Rate = {k/n:.4f}")
            print("Systematic G Matrix:\n", G)
            print("Systematic H Matrix:\n", H)
        except ValueError:
            print(" Validation Error: Please enter a valid integer for r!")
            sys.exit()
            
    elif choice == '3':
        print("\n--- Phase 4: BSC Standalone Test ---")
        try:
            bit_str = input("Enter a binary string to transmit (e.g., 1011001): ")
            if not bit_str or not all(b in ['0', '1'] for b in bit_str):
                print(" Validation Error: Transmission stream must contain ONLY '0' and '1'!")
                sys.exit()
                
            bits = np.array([int(b) for b in bit_str])
            p = float(input("Enter channel error probability P_bar (0 to 0.5): "))
            if p < 0 or p > 0.5:
                print(" Validation Error: Channel error probability (P_bar) must be between 0 and 0.5!")
                sys.exit()
                
            rx = simulate_bsc(bits, p)
            print("Transmitted (TX):", bits)
            print("Received    (RX):", rx)
            print("Errors Injected :", np.sum(bits != rx))
        except ValueError:
            print(" Validation Error: Invalid numeric formatting inside crossover probability!")
            sys.exit()
        
    elif choice == '4':
        print("\n--- Phase 1: Full Chain Input Configuration ---")
        try:
            q = int(input("Enter number of source symbols (q, e.g., 4): "))
            if q <= 1:
                print(" Validation Error: q must be greater than 1!")
                sys.exit()
                
            symbols = [f"S{i+1}" for i in range(q)]
            probabilities = []
            for i in range(q):
                p_val = float(input(f"  Enter probability for {symbols[i]}: "))
                if p_val < 0 or p_val > 1:
                    print(" Validation Error: Probabilities must be in [0, 1] range!")
                    sys.exit()
                probabilities.append(p_val)
                
            if not np.isclose(sum(probabilities), 1.0, atol=1e-5):
                print(f" Validation Error: Sum of probabilities must equal 1.0! (Current sum: {sum(probabilities)})")
                sys.exit()
                
            N = int(input("Enter source sequence length N (e.g., 12000): "))
            L = int(input("Enter extension order L (e.g., 2): "))
            if N <= 0 or L <= 0:
                print(" Validation Error: N and L must be strictly positive values!")
                sys.exit()
            if N % L != 0:
                print(f" Validation Error: Sequence length N ({N}) must be exactly divisible by extension order L ({L}).")
                sys.exit()
                
            r = int(input("Enter Hamming parity bits r (e.g., 3): "))
            if r < 2:
                print(" Validation Error: Parity rows constraint r must be >= 2!")
                sys.exit()
                
            print("\n Simulating full communication chain across SNR values...")
            
            codebook = get_dynamic_huffman_dict(symbols, probabilities, L)
            source_seq = np.random.choice(symbols, size=N, p=probabilities)
            
            binary_stream = ""
            for i in range(0, len(source_seq), L):
                block = tuple(source_seq[i:i+L])
                binary_stream += codebook[block]
            
            tx_bits = np.array([int(b) for b in binary_stream])
            G, H, n, k = generate_hamming_matrices(r)
            
            coded_bits = []
            for i in range(0, len(tx_bits), k):
                block = tx_bits[i:i+k]
                if len(block) < k:
                    block = np.pad(block, (0, k - len(block)), 'constant')
                coded_bits.extend(np.dot(block, G) % 2)
            coded_bits = np.array(coded_bits)
            
            p_errors = np.linspace(0.005, 0.18, 12)
            snr_axis, ber_uncoded, ber_coded = [], [], []
            
            for p_sim in p_errors:
                snr = (1 - p_sim) / p_sim
                snr_axis.append(snr)
                
                rx_uncoded = simulate_bsc(tx_bits, p_sim)
                ber_uncoded.append(np.mean(tx_bits != rx_uncoded))
                
                rx_coded = simulate_bsc(coded_bits, p_sim)
                decoded = decode_hamming(rx_coded, H, n, k)
                
                min_len = min(len(tx_bits), len(decoded))
                ber_coded.append(max(np.mean(tx_bits[:min_len] != decoded[:min_len]), 1e-4))
            
            R = k / n
            low, high = 1e-6, 0.5 - 1e-6
            p_star = 0.1
            for _ in range(50):
                mid = (low + high) / 2
                h_p = -mid * np.log2(mid) - (1 - mid) * np.log2(1 - mid)
                if np.isclose(1 - h_p, R, atol=1e-4):
                    p_star = mid
                    break
                elif (1 - h_p) > R: low = mid
                else: high = mid
            shannon_limit_snr = (1 - p_star) / p_star
            
            plt.figure(figsize=(7, 5))
            plt.yscale('log')
            plt.xscale('log')
            plt.plot(snr_axis, ber_uncoded, 'x-', color='orange', label='Scenario 1: Uncoded System')
            plt.plot(snr_axis, ber_coded, 'o-', color='blue', label=f'Hamming({n},{k}) Coded')
            plt.axvline(x=shannon_limit_snr, color='red', linestyle='--', label=f'Shannon Limit Boundary (SNR={shannon_limit_snr:.2f})')
            plt.title(f'Performance Chart (L={L}, Hamming={n},{k})')
            plt.xlabel('Signal-to-Noise Ratio (SNR) [Log Scale]')
            plt.ylabel('Bit Error Rate (Pe) [Log Scale]')
            plt.grid(True, which="both", ls="--")
            plt.legend()
            plt.show()
            
        except ValueError:
            print(" Validation Error: Input type mismatch! Please check integers/floats requirements.")
            sys.exit()
    else:
        print(" Invalid Option selected!")
        sys.exit()

if __name__ == "__main__":
    main()