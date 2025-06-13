import numpy as np

array = np.random.randint(1,50,size=(5,4))
print("Array:\n",array)

print()

anti_diag = np.fliplr(array).diagonal()
print("Anti Diagonal:",anti_diag)

print()

for a in array:
    print("Maximum element in row",a,"is",a.max())

print()

mean = np.mean(array)
bool = array<=mean
print("Mean:",mean)
print("Array with elements less than or equal to mean:",array[bool])

print()

def numpy_boundary_transversal(A):
    boundary = []
    B = A.transpose()
    for a in A[0, :]:
        if a not in boundary:
            boundary.append(a)
    for a in B[-1, :]:
        if a not in boundary:
            boundary.append(a)
    for a in A[-1, :]:
        if a not in boundary:
            boundary.append(a)
    for a in B[0, :]:
        if a not in boundary:
            boundary.append(a)
    return boundary

print("Boundary Array:",numpy_boundary_transversal(array))