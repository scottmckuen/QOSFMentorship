"""
A quantum circuit for deciding if an integer n is less than a given threshold k is given in the below code.
I implemented it using Cirq.  To compare two k-bit integers, it requires 4 qubit registers of size k (two for
the input, and two anicillas for the output) and a 5th register with k-1 qubits; the bits from the extra
register act as control bits when the higher-order bits match and you need to use lower-order bits to finish 
the comparison.  This is a direct comparison circuit, instead of using an adder and figuring out if the 
difference is negative.  

The n-bit comparator assumes that the two binary-string inputs are zero-padded to have the same length.  
Each pair of bits are compared.  The circuit is built recursively, applying to the (n-1) lower-order bits.
There is a one-bit comparator subcircuit that gets applied to the highest-order bits that haven't been 
used so far.  Two output ancillas are needed because there are three possible states for each bit comparison,
a < b, a > b, or a = b.  If two bits at the current level are equal, then an anti-Toffoli gate on the 5th 
register allows the results from the recursive call to bubble up and overwrite the current level's output ancilla.

Practically (on my laptop), this subroutine is limited to working with 4-bit integers.  It uses
5k-1 qubits to compare two k-bit integers, so the full state grows by a factor of 32 with each increment
in k.  I don't know how much internal optimization Cirq does to control RAM use - there's some tricks
you could do with order-of-operations between matrix and tensor products to try to control the intermediate
size of the program state, but thanks to all the Toffoli gates, there's a lot of entanglement by the middle 
of the circuit and I haven't done an careful analysis.

-----

We use the n-bit comparator as a subroutine in an ordinary classical traversal of the list to identify the 
small elements.  I wanted to use this as the oracle in a Grover search, but I ran into some tricky problems 
and didn't find a complete way through them yet.

1) we are looking for all solutions, not just one.  Multiple solutions takes us out of the straight Grover
search and into amplitude amplification.  I found examples implementing this but have not absorbed them
fully.

2) we don't know the number of solutions ahead of time.  There's a quantum counting algorithm that 
can get this answer, which we then have to use as an input for the amplitude amplification routine.
This is in the same state.

3) We need decide how to handle possible repetitions, because this is a list, not a set.  Also, because this 
is a list, we might want to actually collect the indices instead of just the values.  This seemingly means 
loading the entire array into some kind of QRAM circuit, where we would have a register to store the indices 
and a register to store the values, and these are coupled in a way so the comparator oracle
can compare the values but return the indices.  I can see how to achieve this for individual solutions; I can't
quite see how to do something like "form the superposition of all sublists of the list, then amplify the correct one."

4) I'm not sure how to deterministically achieve the task in a quantum circuit that uses all of these features
without massively blowing up the number of qubits.  That would make the whole thing untestable on my hardware,
so we would have to rely on testing each piece of the algorithm separately on small inputs.  Even the QRAM
addition alone will probably double the qubits required.



"""


from cirq import Circuit, LineQubit, Simulator, measure, X, CCNOT


def anti_toffoli_circuit(circuit, a, b, c):
    """
    Create a circuit that performs the anti-Toffoli operation
    (flip the target qubit if both control qubits are 0).

    @param circuit: a Cirq Circuit
    @param a: first control qubit
    @param b: second control qubit
    @param c: the target qubit

    @return: None, just apply gates to the qubits
    """
    # just flip the inputs, then flip them back after a Toffoli
    circuit.append(X(a))
    circuit.append(X(b))
    circuit.append(CCNOT(a, b, c))
    circuit.append(X(b))
    circuit.append(X(a))


def one_bit_comparator_circuit(circuit, bit1, bit2, ancilla1, ancilla2):
    """
    Create a circuit that compares two bits and stores the result in the ancilla qubits.
    If bit1 > bit2, ancilla1 is set to 1.
    If bit1 < bit2, ancilla2 is set to 1.
    Otherwise, both ancilla qubits are set to 0.

    @param circuit: a Cirq Circuit
    @param bit1:  first bit to compare
    @param bit2:  second bit to compare
    @param ancilla1: a readout qubit, initially in the |0> state
    @param ancilla2: a readout qubit, initially in the |0> state
    @return: None, just apply gates to the qubits to build the circuit
    """
    circuit.append(X(bit2))
    circuit.append(CCNOT(bit1, bit2, ancilla1))
    circuit.append(X(bit1))
    circuit.append(X(bit2))
    circuit.append(CCNOT(bit1, bit2, ancilla2))
    circuit.append(X(bit1))


def n_bit_comparator_circuit(circuit, a_qbs, b_qbs, a_ancillae, b_ancillae, extra_ancillae):
    """
    Create a circuit that compares two bitstrings (as integers) and stores the result
    in the ancilla qubits.  This is done in standard-digit order, so the most significant
    bit is listed first.  The result is stored in the first ancilla qubit of each
    register, a_ancllae[0] and b_ancilla[0].
    a > b if a_ancilla[0] is |1>, b > a if b_ancilla[0] is |1>, and a == b if both are |0>.

    @param circuit: a Cirq Circuit
    @param a_qbs:  list of n qubits representing the first number
    @param b_qbs:  list of n qubits representing the second number
    @param a_ancillae: list of ancilla qubits, as |0>^n, used when an a-bit is bigger
    @param b_ancillae: list of more ancilla qubits, as |0>^n, used when a b-bit is bigger
    @param extra_ancillae:  (n-1) ancilla that control the carry-like operation between the one-bit comparisons

    @return: None, just apply gates to the qubits
    """
    if len(a_qbs) != len(b_qbs):
        raise ValueError("a_qbs and b_qbs must have the same length")
    if len(a_qbs) != len(a_ancillae):
        raise ValueError("a_qbs and a_ancillae must have the same length")
    if len(b_qbs) != len(b_ancillae):
        raise ValueError("b_qbs and b_ancillae must have the same length")
    if len(a_qbs) != len(extra_ancillae) + 1:
        raise ValueError("extra_ancillae must be one fewer than a_qbs")

    one_bit_comparator_circuit(circuit, a_qbs[0], b_qbs[0], a_ancillae[0], b_ancillae[0])

    if len(extra_ancillae) > 0:  # we run out of extra_ancillae just before last step
        # lower-order bits will only be used if the higher-order bits are equal (|xx00>)
        anti_toffoli_circuit(circuit, a_ancillae[0], b_ancillae[0], extra_ancillae[0])

        # recursively call on the lower-order bits
        n_bit_comparator_circuit(circuit,
                                 a_qbs[1:], b_qbs[1:],
                                 a_ancillae[1:], b_ancillae[1:],
                                 extra_ancillae[1:])

        # try to pull any lower-order results up to our higher-order ancilla
        # but this could be blocked by the anti-Toffoli gate result earlier
        circuit.append(CCNOT(a_ancillae[1], extra_ancillae[0], a_ancillae[0]))
        circuit.append(CCNOT(b_ancillae[1], extra_ancillae[0], b_ancillae[0]))

    # if the higher-order bits are equal, the lower-order bits are irrelevant
    # because their results are already in the ancillae after the Toffolis.


def quantum_less_than(a, b):
    """
    Determine if a is less than b.

    @param a: int
    @param b: int
    @return: bool
    """

    n_bits = max(a.bit_length(), b.bit_length())

    # we need n_bits qubits for each of a, b, a_ancilla, and b_ancilla,
    # and 1 fewer "extra" ancilla to go between bit-significance layers
    # (we could also do this as a 5xn grid, which might be more intuitive)
    all_qubits = LineQubit.range(5*n_bits-1)

    # grouping the qubits by purpose (register, ancilla, etc.)
    a_idx = [x for x in range(n_bits)]
    a_qbs = [all_qubits[idx] for idx in a_idx]

    b_idx = [x for x in range(n_bits, 2*n_bits)]
    b_qbs = [all_qubits[idx] for idx in b_idx]

    # for each binary digit, we need two ancilla to store the
    # comparison outcome (three possible results)
    a_ancilla_idx = [x for x in range(2*n_bits, 3*n_bits)]
    a_ancillae = [all_qubits[idx] for idx in a_ancilla_idx]

    b_ancilla_idx = [x for x in range(3*n_bits, 4*n_bits)]
    b_ancillae = [all_qubits[idx] for idx in b_ancilla_idx]

    # between the bit significance layers, we will need an extra ancilla
    # to decide if lower-order comparisons are needed
    extra_ancilla_idx = [x for x in range(4*n_bits, 5*n_bits-1)]
    extra_ancillae = [all_qubits[idx] for idx in extra_ancilla_idx]

    # create the circuit
    circuit = Circuit()
    # initialize the "a" register
    a_binary = f"{a:0{n_bits}b}"
    for i, bit in enumerate(a_binary):
        if bit == '1':
            circuit.append(X(a_qbs[i]))

    # initialize the "b" register
    b_binary = f"{b:0{n_bits}b}"
    for i, bit in enumerate(b_binary):
        if bit == '1':
            circuit.append(X(b_qbs[i]))

    # construct the comparator circuit on the registers and ancillae
    n_bit_comparator_circuit(circuit, a_qbs, b_qbs, a_ancillae, b_ancillae, extra_ancillae)

    # measure the high-bit ancillae for the result
    circuit.append(measure(a_ancillae[0], key='a'))
    circuit.append(measure(b_ancillae[0], key='b'))
    print(circuit)

    print(f"a: {a}, b: {b}")
    print(f"a: {a_binary}, b: {b_binary}")

    simulator = Simulator()
    result = simulator.run(circuit, repetitions=1)

    print(result)
    return result.data['b'][0] == 1


def less_than_k(k: int, array: list[int]) -> list[int]:
    """
    Return a list of elements from the input array that are less than k.
    @param k: a threshold integer
    @param array: a list of integers that may or may not be less than k
    @return: the integers that are less than k, in the order they appear in the input array
    """
    return [x for x in array if quantum_less_than(x, k)]


"""
Here we run the less_than_k function on a sample list of 
integers from the problem description, with a threshold of 7.
"""
scan_this = [4, 9, 11, 14, 1, 13, 6, 15]
threshold = 7
print(f"List to filter: {scan_this}")
print(f"Threshold: {threshold}")
A = less_than_k(threshold, scan_this)
print(f"Results: {A}")


