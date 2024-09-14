import torch

def gram_schmidth(v, w):
    '''
    v, w should be in shape either [b,3] or [b,1,3]

    output:
    - R: [b, 3, 3]
    - v_local: [b, 3] or [b, 1, 3]
    - w_local: [b, 3] or [b, 1, 3]
    '''
    shapes = v.shape
    batch_size = shapes[0]

    v = v.reshape(batch_size, 3)
    w = w.reshape(batch_size, 3)

    # Normalize the first vector
    v_orthonormal = v / (torch.linalg.vector_norm(v,dim=-1, keepdim=True) + 1e-8)

    # Project w onto v_orthonormal and subtract to make w orthogonal to v
    proj = torch.linalg.vecdot(w, v_orthonormal).unsqueeze(-1) * v_orthonormal
    w_orthogonal = w - proj

    # Normalize the second vector
    w_orthonormal = w_orthogonal / (torch.linalg.vector_norm(w_orthogonal, dim=-1, keepdim=True) + 1e-8)


    # Compute the third orthonormal vector using the cross product
    u_orthonormal = torch.linalg.cross(v_orthonormal, w_orthonormal)

    # Construct the rotation matrix R
    R = torch.stack((v_orthonormal, w_orthonormal, u_orthonormal), dim=-1)

    
    RT  = R.transpose(-1, -2)
    # Compute the local frame coordinates
    
    v = v.unsqueeze(-1)
    w = w.unsqueeze(-1)
  
    v_local = torch.matmul(RT,v)
    w_local = torch.matmul(RT, w)
    v_local = v_local.reshape(shapes)
    w_local = w_local.reshape(shapes)
    
    return R, v_local, w_local


torch.manual_seed(1)
v = torch.randn(1, 3)
w = torch.randn(1, 3)

R, v_local, w_local = gram_schmidth(v, w)
print(v_local)
print(w_local)

v = v.reshape(3,1)
w = w.reshape(3,1)

q1 = v 
q2 = w - (w.T@v) / (v.T@v) * v

feat1 = v.T@q1 / torch.sqrt(q1.T@q1)
feat2 = w.T@q1 / torch.sqrt(q1.T@q1)
feat3 = w.T@q2 / torch.sqrt(q2.T@q2)

print(feat1)
print(feat2)
print(feat3)

# if number close
print('if feat1 close? ', torch.allclose(v_local[0,0], feat1.squeeze()))
print('if feat2 close? ', torch.allclose(w_local[0,0], feat2.squeeze()))
print('if feat3 close? ', torch.allclose(w_local[0,1], feat3.squeeze()))
