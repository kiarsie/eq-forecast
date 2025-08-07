def merge_sparse_bins(bounds, catalog, threshold=10):
    lons = catalog.get_longitudes()
    lats = catalog.get_latitudes()
    
    def count_events(b):
        in_bin = (
            (lons >= b[0]) & (lons < b[1]) &
            (lats >= b[2]) & (lats < b[3])
        )
        return in_bin.sum()
    
    bounds = list(bounds)
    counts = [count_events(b) for b in bounds]

    merged = True
    while merged:
        merged = False
        for i in range(len(bounds)):
            if counts[i] >= threshold:
                continue
            for j in range(len(bounds)):
                if i == j or counts[j] < threshold:
                    continue
                b1, b2 = bounds[i], bounds[j]
                # Check for shared border (naïve)
                if (b1[0] == b2[1] or b1[1] == b2[0]) and (b1[2] == b2[2] and b1[3] == b2[3]):
                    new_bin = (
                        min(b1[0], b2[0]), max(b1[1], b2[1]),
                        b1[2], b1[3]
                    )
                elif (b1[2] == b2[3] or b1[3] == b2[2]) and (b1[0] == b2[0] and b1[1] == b2[1]):
                    new_bin = (
                        b1[0], b1[1],
                        min(b1[2], b2[2]), max(b1[3], b2[3])
                    )
                else:
                    continue
                bounds.pop(max(i, j))
                bounds.pop(min(i, j))
                bounds.append(new_bin)
                counts = [count_events(b) for b in bounds]
                merged = True
                break
            if merged:
                break
    return bounds
