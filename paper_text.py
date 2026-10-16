def merge_overlapping(left,right):
    max_size = min(len(left),len(right))

    for size in range(max_size,0,-1):
        if left[-size:] == right[:size]:
            return left + right[size:]

    return left+right
