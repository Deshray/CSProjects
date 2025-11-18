# Get list from user
listnum = int(input("How many values do you want in your list? :"))
lst = []

for i in range(listnum):
    listval = int(input(f"Enter value {i+1} for the list: "))
    lst.append(listval)

# Sort the list (required for binary search)
lst.sort()
print("Sorted list:", lst)

val = int(input("Which item in the list do you want to search for? :"))

low = 0
high = len(lst) - 1  # Fixed: should be len(lst) - 1, not len(lst)
found = False

while low <= high:  # Fixed: should be <=, not !=
    mid = (low + high) // 2  # Fixed: should be integer division //
    
    if lst[mid] == val:
        print(f"Value {val} found at index {mid}!")
        found = True
        break
    elif lst[mid] < val:
        low = mid + 1
    else:
        high = mid - 1

if not found:
    print(f"Value {val} not found in the list.")