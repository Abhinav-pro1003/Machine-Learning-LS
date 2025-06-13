import numpy as np

array = np.random.uniform(0, 10, size = 20)
arr = np.round(array, 2)
print("Rounded Array:",arr)

print()

print("Minimum:",arr.min())
print("Maximum",arr.max())
x = np.median(arr)
print("Median",f"{x:.2f}")

print()

arr = [x*x if x < 5.00 else x for x in arr]
arr = np.round(arr, 2)
print(arr)

print()

def numpy_alternate_sort(array):
    array = np.sort(array)
    list = []
    a=0
    while a<=(np.size(array)/2-1):
        list.append(array[a])
        a+=1
        list.append(array[-a])
    if np.size(array)%2==1:
        list.append(array[a])
    return list
print("Sorted Array:",numpy_alternate_sort(arr))