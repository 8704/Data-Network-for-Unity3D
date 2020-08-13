using UnityEngine;

[System.Serializable]
public class Plane {
    [HideInInspector] public Vector3[] verts;
    [HideInInspector] public Vector2[] uvs;
    [HideInInspector] public int[] tris;
    Mesh mesh;
    MeshFilter meshfilter;
    Material material;
    public void Create(float Size, int xSize, int zSize, MeshFilter mf, float height = 0, bool vertical = false) {
        verts = new Vector3[(xSize + 1) * (zSize + 1)];
        uvs = new Vector2[verts.Length];
        for (int i = 0, z = 0; z <= zSize; z++) {
            for (int x = 0; x <= xSize; x++) {
                if (vertical)
                    verts[i] = new Vector3(x * (Size / xSize), z * (Size / zSize), height);
                else
                    verts[i] = new Vector3(x * (Size / xSize), height, z * (Size / zSize));
                uvs[i] = new Vector2(x * (1f / xSize), z * (1f / zSize));
                i++;
            }
        }

        tris = new int[xSize * zSize * 6];
        int vert = 0;
        int tri = 0;
        for (int z = 0; z < zSize; z++) {
            for (int x = 0; x < xSize; x++) {
                tris[tri + 0] = vert;
                tris[tri + 1] = vert + xSize + 1;
                tris[tri + 2] = vert + 1;
                tris[tri + 3] = vert + 1;
                tris[tri + 4] = vert + xSize + 1;
                tris[tri + 5] = vert + xSize + 2;
                vert++;
                tri += 6;
            }
            vert++;
        }

        if (!mf.sharedMesh) {
            mesh = new Mesh();
            mf.sharedMesh = mesh;
        }
        meshfilter = mf;
        mesh = mf.sharedMesh;

        mesh.Clear();
        mesh.vertices = verts;
        mesh.uv = uvs;
        mesh.triangles = tris;
        mesh.RecalculateNormals();
        material = Resources.Load<Material>("VertexColor");
        meshfilter.GetComponent<Renderer>().material = material;
    }

    public void AddColor(Color color) {
        Color[] colorArray = new Color[verts.Length];
        for (int a = 0; a < colorArray.Length; a++) {
            colorArray[a] = color;
        }
        mesh.colors = colorArray;

    }

    public void AddUniqueMaterial(Material material) {
        meshfilter.GetComponent<Renderer>().material = material;
    }
}