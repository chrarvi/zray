import numpy as np


def generate_d20(filename="d20.txt"):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array(
        [
            [0, 1, phi],
            [0, -1, phi],
            [0, 1, -phi],
            [0, -1, -phi],
            [1, phi, 0],
            [-1, phi, 0],
            [1, -phi, 0],
            [-1, -phi, 0],
            [phi, 0, 1],
            [-phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, -1],
        ],
        dtype=float,
    )
    verts /= np.linalg.norm(verts[0])
    faces = np.array(
        [
            [0, 1, 8],
            [0, 1, 9],
            [0, 4, 5],
            [0, 4, 8],
            [0, 5, 9],
            [1, 6, 7],
            [1, 6, 8],
            [1, 7, 9],
            [2, 3, 10],
            [2, 3, 11],
            [2, 4, 5],
            [2, 4, 10],
            [2, 5, 11],
            [3, 6, 7],
            [3, 6, 10],
            [3, 7, 11],
            [4, 8, 10],
            [5, 9, 11],
            [6, 8, 10],
            [7, 9, 11],
        ]
    )

    with open(filename, "w") as f:
        f.write(f"{len(faces)}\n")
        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            normal /= np.linalg.norm(normal)
            for v in [v0, v1, v2]:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                f.write(f"{normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            f.write("\n")


if __name__ == "__main__":
    path = "icosahedron.txt"
    generate_d20(path)
    print(f"icosahedron mesh file written to {path}")
