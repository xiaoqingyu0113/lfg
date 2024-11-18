import sympy as sp
import inspect

# Define symbols for inputs
v1, v2, v3 = sp.symbols('v1 v2 v3')
w1, w2, w3 = sp.symbols('w1 w2 w3')
v = sp.Matrix([v1, v2, v3])
w = sp.Matrix([w1, w2, w3])

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

    # return R, v_local, w_local

    # Stack R, v_local, and w_local into a single vector
    stacked_output = sp.Matrix.vstack(R.reshape(9, 1), v_local, w_local)
    
    return stacked_output

# Get outputs from gs function
stacked_output = gs(v, w)

# Compute the Jacobian of R, v_local, w_local with respect to v and w
# Flatten R to make it easier to compute the Jacobian

# Compute Jacobians
J_R_v = stacked_output.jacobian([v1,v2,v3,w1,w2,w3])   # Jacobian of R with respect to v
print(J_R_v.shape)
# lambdify the Jacob
J_R_v = sp.lambdify((v,w), J_R_v, 'numpy')
J_code = inspect.getsource(J_R_v)


filename = 'lfg/derive.py'
with open(filename, 'w') as f:
    f.write('from numpy import array\n')

with open(filename, 'a') as f:
    J_code = J_code.replace('_lambdifygenerated', 'dJ')
    f.write(J_code)