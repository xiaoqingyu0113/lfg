import sympy as sp

# Define matrices and biases
recode_weight = sp.MatrixSymbol('recode_weight', 3, 3)
recode_bias = sp.MatrixSymbol('recode_bias', 3, 1)
layer1_weight = sp.MatrixSymbol('layer1_weight', 32, 3)
layer1_bias = sp.MatrixSymbol('layer1_bias', 32, 1)
layer2_weight = sp.MatrixSymbol('layer2_weight', 32, 32)
layer2_bias = sp.MatrixSymbol('layer2_bias', 32, 1)
dec_0_weight = sp.MatrixSymbol('dec_0_weight', 128, 32)
dec_0_bias = sp.MatrixSymbol('dec_0_bias', 128, 1)
dec_2_weight = sp.MatrixSymbol('dec_2_weight', 3, 128)
dec_2_bias = sp.MatrixSymbol('dec_2_bias', 3, 1)
bias = sp.MatrixSymbol('bias', 3, 1)
v = sp.Matrix(sp.MatrixSymbol('v', 3, 1))  # Convert to Matrix for operations
w = sp.Matrix(sp.MatrixSymbol('w', 3, 1))  # Convert to Matrix for operations

def gs(v, w):
    # Normalize v
    v_orthonormal = v / (sp.sqrt(v.dot(v)) + sp.Float(1e-6, 6))
    
    # Project w onto v_orthonormal and calculate w_orthogonal
    proj = w.dot(v_orthonormal) * v_orthonormal
    w_orthogonal = w - proj
    
    # Normalize w_orthogonal
    w_orthonormal = w_orthogonal / (sp.sqrt(w_orthogonal.dot(w_orthogonal)) + sp.Float(1e-6, 6))
    
    # Calculate the orthogonal vector using the cross product
    u_orthonormal = v_orthonormal.cross(w_orthonormal)
    
    # Create the rotation matrix R as a SymPy Matrix
    R = sp.Matrix.hstack(v_orthonormal, w_orthonormal, u_orthonormal)
    
    # Transform v and w to the local coordinates
    v_local = R.T * v
    w_local = R.T * w

    return R, v_local, w_local

# Perform the matrix multiplication and add recode_bias
w_transformed = sp.Matrix(recode_weight) * w + sp.Matrix(recode_bias)

# Call the gs function
R, v_local, w_local = gs(v, w_transformed)

# Calculate the feature vector
feat = sp.Matrix([[v_local[0]], [w_local[0]], [w_local[1]]])

# Calculate the hidden layer activations

h = sp.Matrix(layer1_weight * feat + layer1_bias)

h.applyfunc(lambda x: sp.Piecewise((x, x > 0), (0, True)))
h2 = sp.Matrix(layer2_weight * h + layer2_bias)
h2.applyfunc(lambda x: sp.Piecewise((x, x > 0), (0, True)))
h2 = h2.multiply_elementwise(h) + h


# Calculate the output+
y = dec_0_weight * h2 + dec_0_bias
# y.applyfunc(lambda x: sp.Piecewise((x, x > 0), (0, True)))
y = dec_2_weight * y + dec_2_bias
y = R * y
y = y + sp.Matrix(bias)

