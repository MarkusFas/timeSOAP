

# positions can be anything compatible with numpy's ndarray
positions = [
    (0, 0, 0),
    (0, 1.3, 1.3),
]
box = 3.2 * np.eye(3)

calculator = NeighborList(cutoff=4.2, full_list=True)
i, j, S, d = calculator.compute(
    points=points,
    box=box,
    periodic=True,
    quantities="ijSd"
)

