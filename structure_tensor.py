 # adapted to dask from https://github.com/Skielex/structure-tensor/blob/master/structure_tensor/st3d.py
# replaced all instances of o'ut = ...' in numpy as this is not my style and it also seems cause problems with dask
import dask.array as lib
import dask_image.ndfilters
    

def structure_tensor_3d(volume, sigma, rho, truncate = 4):
    # Computing derivatives (scipy implementation truncates filter at 4 sigma).

    # there might be an issue when feeding non-4D stacks, to be tested
    
    Vx = dask_image.ndfilters.gaussian_filter(volume, sigma, order=(0, 0, 1, 0), mode="nearest", truncate=truncate)  # type: ignore
    Vy = dask_image.ndfilters.gaussian_filter(volume, sigma, order=(0, 1, 0, 0), mode="nearest", truncate=truncate)  # type: ignore
    Vz = dask_image.ndfilters.gaussian_filter(volume, sigma, order=(1, 0, 0, 0), mode="nearest", truncate=truncate)  # type: ignore
    
    S = lib.empty((6,) + volume.shape, dtype=volume.dtype)

    # Integrating elements of structure tensor (scipy uses sequence of 1D).
    lib.multiply(Vx, Vx, out=S[0])
    S[0] = dask_image.ndfilters.gaussian_filter(S[0], rho, mode="nearest",truncate=truncate)
    S[1] = lib.multiply(Vy, Vy)
    S[1] = dask_image.ndfilters.gaussian_filter(S[1], rho, mode="nearest", truncate=truncate)
    S[2] = lib.multiply(Vz, Vz)
    S[2] = dask_image.ndfilters.gaussian_filter(S[2], rho, mode="nearest", truncate=truncate)
    S[3] = lib.multiply(Vx, Vy)
    S[3] = dask_image.ndfilters.gaussian_filter(S[3], rho, mode="nearest", truncate=truncate)
    S[4] = lib.multiply(Vx, Vz)
    S[4] = dask_image.ndfilters.gaussian_filter(S[4], rho, mode="nearest", truncate=truncate)
    S[5] = lib.multiply(Vy, Vz)
    S[5] = dask_image.ndfilters.gaussian_filter(S[5], rho, mode="nearest", truncate=truncate)

    return S

def eigenvalues(S):
    #structure tensor eigenvalues
    
    # Flatten S.
    input_shape = S.shape
    S = S.reshape(6, -1)   
    # Create v vector.
    v = lib.array([[2 * lib.pi / 3], [4 * lib.pi / 3]], dtype=S.dtype)
    # Computing eigenvalues.
    # Allocate vec and val. We will use them for intermediate computations as well. RF: Not sure if this is the best way in dask ...
    # val = lib.empty((3,) + S.shape[1:], dtype=S.dtype)
    # vec = lib.empty((3,) + S.shape[1:], dtype=S.dtype)
    # tmp = lib.empty((4,) + S.shape[1:], dtype=S.dtype)
    # B03 = val
    # B36 = vec

    # Views for B.
    # B0 = B03[0]
    # B1 = B03[1]
    # B2 = B03[2]
    # B3 = B36[0]
    # B4 = B36[1]
    # B5 = B36[2]

    # Compute q, mean of diagonal. We need to use q multiple times later.
    # Using lib.mean has precision issues.
    q = lib.add(S[0], S[1]) #, out=tmp[0])
    q += S[2]
    q /= 3

    # Compute S minus q. Insert it directly into B where it'll stay.
    Sq = lib.subtract(S[:3], q) #, out=B03)
    B03 = Sq

    # Compute s, off-diagonal elements. Store in part of B not yet used.
    s = lib.einsum("ij,ij->j", S[3:], S[3:])#, out=tmp[1])
    s *= 2

    # Compute p.
    p = lib.einsum("ij,ij->j", Sq, Sq)#, out=tmp[2])
    # del Sq  # Last use of Sq. RF: copied. I don't think deleting a dask array before computation is a good idea --> commented out
    p += s         #RF: I don't understand this syntax. To be checked if making problems
    p *= 1 / 6
    p = lib.sqrt(p)

    # Compute inverse p, while avoiding 0 division.
    # Reuse s allocation and delete s variable. RF: Why tho?
    p_inv = s  
    # del s      #RF: same comment as above. don't delete in dask
    p_inv[:] = 0 #RF this empties s which was in tmp --> better rewrite whole function without the memory allocation not needed in dask to avoid confusion
    lib.divide(1, p, out=p_inv, where=p != 0)  #RF: here the out= is probably needed

    # Compute d, determinant of B.
    d = lib.prod(B03, axis=0)#, out=tmp[3])

    # Reuse allocation for p_inv and delete variable.
    d_tmp = p_inv
    # del p_inv
    # Computation of d.
    lib.multiply(B2, B3, d_tmp)
    d_tmp *= B3
    d -= d_tmp
    lib.multiply(B4, B4, out=d_tmp)
    d_tmp *= B1
    d -= d_tmp
    lib.prod(B36, axis=0, out=d_tmp)
    d_tmp *= 2
    d += d_tmp
    lib.multiply(B5, B5, out=d_tmp)
    d_tmp *= B0
    d -= d_tmp
    d *= 0.5
    # Ensure -1 <= d/2 <= 1.
    lib.clip(d, -1, 1, out=d)

    # Compute phi. Beware that we reuse d variable!
    phi = d
    # del d
    phi = lib.arccos(phi, out=phi)
    phi /= 3

    
    # Compute val, ordered eigenvalues. Resuing B allocation.
    # del B03, B36, B0, B1, B2, B3, B4, B5

    lib.add(v, phi[lib.newaxis], out=val[:2])
    val[2] = phi
    lib.cos(val, out=val)
    p *= 2
    val *= p
    val += q

    # Computing eigenvectors -- only one vs. all three in orignal code
    l = val[0]
    vec_tmp = tmp[2]

    # Compute vec. The tmp variable can be reused. RF: revise this variable re-use. could be a problem in dask
    # RF: I don't think we need the eigenvector here. Test once and then comment out

    # u = S[4] * S[5] - (S[2] - l) * S[3]
    u = lib.subtract(S[2], l, out=vec[0])
    lib.multiply(u, S[3], out=u)
    u_tmp = lib.multiply(S[4], S[5], out=tmp[3])
    lib.subtract(u_tmp, u, out=u)
    # Put values of u into vector 2 aswell.

    # v = S[3] * S[5] - (S[1] - l) * S[4]
    v = lib.subtract(S[1], l, out=vec_tmp)
    lib.multiply(v, S[4], out=v)
    v_tmp = lib.multiply(S[3], S[5], out=tmp[3])
    lib.subtract(v_tmp, v, out=v)

    # w = S[3] * S[4] - (S[0] - l) * S[5]
    w = lib.subtract(S[0], l, out=vec[2])
    lib.multiply(w, S[5], out=w)
    w_tmp = lib.multiply(S[3], S[4], out=tmp[3])
    lib.subtract(w_tmp, w, out=w)

    vec[1] = u
    lib.multiply(u, v, out=vec[0])
    u = vec[1]
    lib.multiply(u, w, out=vec[1])
    lib.multiply(v, w, out=vec[2])

    # Normalizing -- depends on number of vectors. RF: here only one
    l = lib.einsum("ij,ij->j", vec, vec, out=vec_tmp)

    lib.sqrt(l, out=l)
    vec /= l

    val = val.reshape(val.shape[:-1] + input_shape[1:])
    vec = vec.reshape(vec.shape[:-1] + input_shape[1:])

    return val, vec