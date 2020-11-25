import os, sys, trimesh, matplotlib.pyplot as pyplot, numpy as np, time, random, progressbar, json
from plyfile import PlyData, PlyElement
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.6f')

from subprocess import call
from collections import deque
from imageio import imread


colors = [[0, 0, 1], [1, 0, 0], [0, 1, 0],
          [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
          [0.3, 0.6, 0], [0.6, 0, 0.3], [0.3, 0, 0.6],
          [0.6, 0.3, 0], [0.3, 0, 0.6], [0.6, 0, 0.3],
          [0.8, 0.2, 0.5]]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0] < num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b, :, :]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert (vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a, b, c] == 1:
                    points.append(np.array([a, b, c]))
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)
    return points


def point_cloud_to_volume_v2_batch(point_clouds, vsize=12, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxVxVxVxnum_samplex3
        Added on Feb 19
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume_v2(point_clouds[b, :, :], vsize, radius, num_sample)
        vol_list.append(np.expand_dims(vol, 0))
    return np.concatenate(vol_list, 0)


def point_cloud_to_volume_v2(points, vsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is vsize*vsize*vsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each voxel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    vol = np.zeros((vsize, vsize, vsize, num_sample, 3))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n, :])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n, :])

    for i in range(vsize):
        for j in range(vsize):
            for k in range(vsize):
                if (i, j, k) not in loc2pc:
                    vol[i, j, k, :, :] = np.zeros((num_sample, 3))
                else:
                    pc = loc2pc[(i, j, k)]  # a list of (3,) arrays
                    pc = np.vstack(pc)  # kx3
                    # Sample/pad to num_sample points
                    if pc.shape[0] > num_sample:
                        pc = random_sampling(pc, num_sample, False)
                    elif pc.shape[0] < num_sample:
                        pc = np.lib.pad(pc, ((0, num_sample - pc.shape[0]), (0, 0)), 'edge')
                    # Normalize
                    pc_center = (np.array([i, j, k]) + 0.5) * voxel - radius
                    pc = (pc - pc_center) / voxel  # shift and scale
                    vol[i, j, k, :, :] = pc
    return vol


def point_cloud_to_image_batch(point_clouds, imgsize, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxIxIxnum_samplex3
        Added on Feb 19
    """
    img_list = []
    for b in range(point_clouds.shape[0]):
        img = point_cloud_to_image(point_clouds[b, :, :], imgsize, radius, num_sample)
        img_list.append(np.expand_dims(img, 0))
    return np.concatenate(img_list, 0)


def point_cloud_to_image(points, imgsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is imgsize*imgsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each pixel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    img = np.zeros((imgsize, imgsize, num_sample, 3))
    pixel = 2 * radius / float(imgsize)
    locations = (points[:, 0:2] + radius) / pixel  # Nx2
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n, :])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n, :])
    for i in range(imgsize):
        for j in range(imgsize):
            if (i, j) not in loc2pc:
                img[i, j, :, :] = np.zeros((num_sample, 3))
            else:
                pc = loc2pc[(i, j)]
                pc = np.vstack(pc)
                if pc.shape[0] > num_sample:
                    pc = random_sampling(pc, num_sample, False)
                elif pc.shape[0] < num_sample:
                    pc = np.lib.pad(pc, ((0, num_sample - pc.shape[0]), (0, 0)), 'edge')
                pc_center = (np.array([i, j]) + 0.5) * pixel - radius
                pc[:, 0:2] = (pc[:, 0:2] - pc_center) / pixel
                img[i, j, :, :] = pc
    return img


# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


def write_ply_color(points, labels, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))

    vertex = []
    # colors = [pyplot.cm.jet(i / float(num_classes)) for i in range(num_classes)]
    colors = [colormap(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x * 255) for x in c]
        vertex.append((points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)
    return colors


def merge_mesh_with_color(meshes):
    face_colors = [mesh.visual.face_colors for mesh in meshes]
    vertex_colors = [mesh.visual.vertex_colors for mesh in meshes]
    vertice_list = [mesh.vertices for mesh in meshes]
    faces_list = [mesh.faces for mesh in meshes]

    faces_offset = np.cumsum([v.shape[0] for v in vertice_list])
    faces_offset = np.insert(faces_offset, 0, 0)[:-1]

    vertices = np.vstack(vertice_list)
    faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)])
    vertex_colors = np.vstack(vertex_colors)
    face_colors = np.vstack(face_colors)
    # print(vertex_colors.shape, faces.shape, vertices.shape)
    # exit(0)
    merged_meshes = trimesh.Trimesh(vertices, faces, face_colors=face_colors, vertex_colors=vertex_colors)
    return merged_meshes


def write_ply_bbox_color(vertices, vertex_colors, edges, edge_colors, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """

    vertex = []
    for i in range(len(vertices)):
        vertex.append((vertices[i, 0], vertices[i, 1], vertices[i, 2], vertex_colors[i, 0],
                       vertex_colors[i, 1], vertex_colors[i, 2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    edge = []
    for i in range(len(edges)):
        edge.append((edges[i, 0], edges[i, 1], edge_colors[i, 0], edge_colors[i, 1], edge_colors[i, 2]))
    edge = np.array(edge,
                    dtype=[('vertex1', 'i4'), ('vertex2', 'i4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    e1 = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    e2 = PlyElement.describe(edge, 'edge', comments=['edges'])
    PlyData([e1, e2], text=True).write(filename)


def write_bbox_color_json(scene_bbox, label, out_filename, num_classes=None, colormap=pyplot.cm.jet):
    labels = label.astype(int)
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))
    colors = [colormap(i / float(num_classes)) for i in range(num_classes)]
    used_color = {}
    ret = []
    for i, box in enumerate(scene_bbox):
        c = colors[label[i]]
        c = (np.array(c) * 255).astype(np.uint8)
        item_i = [float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]),
                  int(c[0]), int(c[1]), int(c[2])]
        used_color[label[i]] = c
        #item_i = [str(_) for _ in item_i]
        ret.append(item_i)
    with open(out_filename, 'w') as f:
        json.dump(ret, f)
    return used_color


def write_bbox_color(scene_bbox, label, out_filename, num_classes=None, colormap=pyplot.cm.jet, edge=False):
    """Export scene bbox to meshes
    Args:
        scene_bbox: (N x 6 numpy array): xyz pos of center and 3 lengths
        out_filename: (string) filename

    Note:
        To visualize the boxes in MeshLab.
        1. Select the objects (the boxes)
        2. Filters -> Polygon and Quad Mesh -> Turn into Quad-Dominant Mesh
        3. Select Wireframe view.
    """

    labels = label.astype(int)
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))

    def convert_box_to_trimesh_fmt(box, color):
        ctr = box[:3]
        lengths = box[3:]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        mesh = trimesh.creation.box(lengths, trns)
        color = np.array(color) * 255
        face_colors = np.array([color] * mesh.faces.shape[0], np.uint8)
        vertex_colors = np.array([color] * mesh.vertices.shape[0], np.uint8)

        #print(face_colors, vertex_colors, box_trimesh_fmt.vertices, box_trimesh_fmt.faces)
        #exit(0)
        box_visual = trimesh.visual.create_visual(
                vertex_colors=vertex_colors,
                face_colors=face_colors,
                mesh=mesh)
        mesh.visual = box_visual

        # print(edges.shape)
        # exit(0)
        # print(box_trimesh_fmt.visual.face_colors)
        #print(face_colors)
        #print(box_visual.__dict__)
        #print(box_trimesh_fmt.visual.__dict__)
        #exit(0)
        #, facecolors=color, vertex_color=color)
        #print(box_trimesh_fmt.__dict__)
        #exit(0)
        return mesh

    colors = [colormap(i / float(num_classes)) for i in range(num_classes)]
    scene = []
    ret = []
    for i, box in enumerate(scene_bbox):
        ret.append(colors[label[i]])
        scene.append(convert_box_to_trimesh_fmt(box, colors[label[i]]))
    mesh = merge_mesh_with_color(scene)
    if edge:
        sharp = mesh.face_adjacency_angles > np.radians(40)
        edges = mesh.face_adjacency_edges[sharp]
        assert edges.shape[0] % 12 == 0
        edge_colors = mesh.visual.vertex_colors[edges[:, 0]]
        #print(edges.shape, edge_colors.shape)
        #exit(0)

        write_ply_bbox_color(mesh.vertices, mesh.visual.vertex_colors, edges, edge_colors, out_filename)
    else:
        trimesh.exchange.export.export_mesh(mesh, out_filename, file_type='ply')
    #print(mesh_list.visual.mesh.visual.__dict__)
    # save to ply file
    # ply = trimesh.exchange.ply.export_ply(mesh_list, encoding='ascii')
    #trimesh.exchange.export.export_mesh(mesh_list, out_filename, file_type='ply') #, encoding='ascii')
    # print(ply)
    # exit(0)
    # out_filename
    return ret


def write_ply_rgb(points, colors, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i, :]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    pyplot.savefig(output_filename)


def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)


# ----------------------------------------
# Simple Point manipulations
# ----------------------------------------
def rotate_point_cloud(points, rotation_matrix=None):
    """ Input: (n,3), Output: (n,3) """
    # Rotate in-place around Z axis.
    if rotation_matrix is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
        sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points - ctr, rotation_matrix) + ctr
    return rotated_data, rotation_matrix


def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


# ----------------------------------------
# BBox
# ----------------------------------------
def bbox_corner_dist_measure(crnr1, crnr2):
    """ compute distance between box corners to replace iou
    Args:
        crnr1, crnr2: Nx3 points of box corners in camera axis (y points down)
        output is a scalar between 0 and 1
    """

    dist = sys.maxsize
    for y in range(4):
        rows = ([(x + y) % 4 for x in range(4)] + [4 + (x + y) % 4 for x in range(4)])
        d_ = np.linalg.norm(crnr2[rows, :] - crnr1, axis=1).sum() / 8.0
        if d_ < dist:
            dist = d_

    u = sum([np.linalg.norm(x[0, :] - x[6, :]) for x in [crnr1, crnr2]]) / 2.0

    measure = max(1.0 - dist / u, 0)
    print(measure)

    return measure


def point_cloud_to_bbox(points):
    """ Extract the axis aligned box from a pcl or batch of pcls
    Args:
        points: Nx3 points or BxNx3
        output is 6 dim: xyz pos of center and 3 lengths
    """
    which_dim = len(points.shape) - 2  # first dim if a single cloud and second if batch
    mn, mx = points.min(which_dim), points.max(which_dim)
    lengths = mx - mn
    cntr = 0.5 * (mn + mx)
    return np.concatenate([cntr, lengths], axis=which_dim)


def write_bbox(scene_bbox, out_filename):
    """Export scene bbox to meshes
    Args:
        scene_bbox: (N x 6 numpy array): xyz pos of center and 3 lengths
        out_filename: (string) filename

    Note:
        To visualize the boxes in MeshLab.
        1. Select the objects (the boxes)
        2. Filters -> Polygon and Quad Mesh -> Turn into Quad-Dominant Mesh
        3. Select Wireframe view.
    """

    def convert_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')

    return


def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """

    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')

    return


def write_oriented_bbox_camera_coord(scene_bbox, out_filename):
    """Export oriented (around Y axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Y axis.
            Z forward, X rightward, Y downward. heading angle of positive X is 0,
            heading angle of negative Z is 90 degrees.
        out_filename: (string) filename
    """

    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3, 3))
        rotmat[1, 1] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0, :] = np.array([cosval, 0, sinval])
        rotmat[2, :] = np.array([-sinval, 0, cosval])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')

    return


def write_lines_as_cylinders(pcl, filename, rad=0.005, res=64):
    """Create lines represented as cylinders connecting pairs of 3D points
    Args:
        pcl: (N x 2 x 3 numpy array): N pairs of xyz pos
        filename: (string) filename for the output mesh (ply) file
        rad: radius for the cylinder
        res: number of sections used to create the cylinder
    """
    scene = trimesh.scene.Scene()
    for src, tgt in pcl:
        # compute line
        vec = tgt - src
        M = trimesh.geometry.align_vectors([0, 0, 1], vec, False)
        vec = tgt - src  # compute again since align_vectors modifies vec in-place!
        M[:3, 3] = 0.5 * src + 0.5 * tgt
        height = np.sqrt(np.dot(vec, vec))
        scene.add_geometry(trimesh.creation.cylinder(radius=rad, height=height, sections=res, transform=M))
    mesh_list = trimesh.util.concatenate(scene.dump())
    trimesh.io.export.export_mesh(mesh_list, '%s.ply' % (filename), file_type='ply')




def normalize_pts(pts):
    out = np.array(pts, dtype=np.float32)
    center = np.mean(out, axis=0)
    out -= center
    scale = np.sqrt(np.max(np.sum(out ** 2, axis=1)))
    out /= scale
    return out


def load_obj(fn, no_normal=False):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = [];
    normals = [];
    faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('vn '):
            normals.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    mesh = dict()
    mesh['faces'] = np.vstack(faces)
    mesh['vertices'] = np.vstack(vertices)

    if (not no_normal) and (len(normals) > 0):
        assert len(normals) == len(vertices), 'ERROR: #vertices != #normals'
        mesh['normals'] = np.vstack(normals)

    return mesh


def export_obj_submesh_label(obj_fn, label_fn):
    fin = open(obj_fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    face_ids = [];
    cur_id = 0;
    for line in lines:
        if line.startswith('f '):
            face_ids.append(cur_id)
        elif line.startswith('g '):
            cur_id += 1

    fout = open(label_fn, 'w')
    for i in range(len(face_ids)):
        fout.write('%d\n' % face_ids[i])
    fout.close()


def load_obj_with_submeshes(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = [];
    submesh_id = -1;
    submesh_names = [];
    faces = dict();
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces[submesh_id].append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))
        elif line.startswith('g '):
            submesh_names.append(line.split()[1])
            submesh_id += 1
            faces[submesh_id] = []

    vertice_arr = np.vstack(vertices)

    mesh = dict()
    mesh['names'] = submesh_names
    mesh['tot'] = submesh_id + 1
    out_vertices = dict()
    out_faces = dict()
    for i in range(submesh_id + 1):
        data = np.vstack(faces[i]).astype(np.int32)

        out_vertice_ids = np.array(list(set(data.flatten())), dtype=np.int32) - 1
        vertice_map = {out_vertice_ids[x] + 1: x + 1 for x in range(len(out_vertice_ids))}
        out_vertices[i] = vertice_arr[out_vertice_ids, :]

        data = np.vstack(faces[i])
        cur_out_faces = np.zeros(data.shape, dtype=np.float32)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                cur_out_faces[x, y] = vertice_map[data[x, y]]
        out_faces[i] = cur_out_faces

    mesh['vertices'] = out_vertices
    mesh['faces'] = out_faces

    return mesh


def load_off(fn):
    fin = open(fn, 'r')
    line = fin.readline()
    line = fin.readline()
    num_vertices = int(line.split()[0])
    num_faces = int(line.split()[1])

    vertices = np.zeros((num_vertices, 3)).astype(np.float32)
    for i in range(num_vertices):
        vertices[i, :] = np.float32(fin.readline().split())

    faces = np.zeros((num_faces, 3)).astype(np.int32)
    for i in range(num_faces):
        faces[i, :] = np.int32(fin.readline().split()[1:]) + 1

    fin.close()

    mesh = dict()
    mesh['faces'] = faces
    mesh['vertices'] = vertices

    return mesh


def rotate_pts(pts, theta=0, phi=0):
    rotated_data = np.zeros(pts.shape, dtype=np.float32)

    # rotate along y-z axis
    rotation_angle = phi / 90 * np.pi / 2
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, sinval],
                                [0, -sinval, cosval]])
    rotated_pts = np.dot(pts, rotation_matrix)

    # rotate along x-z axis
    rotation_angle = theta / 360 * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_pts = np.dot(rotated_pts, rotation_matrix)
    return rotated_pts


def load_pts(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines],
                       dtype=np.float32)
        return pts


def load_pts_nor(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines],
                       dtype=np.float32)
        nor = np.array([[float(line.split()[3]), float(line.split()[4]), float(line.split()[5])] for line in lines],
                       dtype=np.float32)
        return pts, nor


def load_label(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        label = np.array([int(line) for line in lines], dtype=np.int32)
        return label


def export_obj(out, v, f):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))


def export_label(out, label):
    with open(out, 'w') as fout:
        for i in range(label.shape[0]):
            fout.write('%d\n' % label[i])


def export_pts(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))


def export_pts_with_normal(out, v, n):
    assert v.shape[0] == n.shape[0], 'v.shape[0] != v.shape[0]'

    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], n[i, 0], n[i, 1], n[i, 2]))


def export_ply(out, v):
    with open(out, 'w') as fout:
        fout.write('ply\n');
        fout.write('format ascii 1.0\n');
        fout.write('element vertex ' + str(v.shape[0]) + '\n');
        fout.write('property float x\n');
        fout.write('property float y\n');
        fout.write('property float z\n');
        fout.write('end_header\n');

        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))


def export_ply_with_label(out, v, l):
    num_colors = len(colors)
    with open(out, 'w') as fout:
        fout.write('ply\n');
        fout.write('format ascii 1.0\n');
        fout.write('element vertex ' + str(v.shape[0]) + '\n');
        fout.write('property float x\n');
        fout.write('property float y\n');
        fout.write('property float z\n');
        fout.write('property uchar red\n');
        fout.write('property uchar green\n');
        fout.write('property uchar blue\n');
        fout.write('end_header\n');

        for i in range(v.shape[0]):
            cur_color = colors[l[i] % num_colors]
            fout.write('%f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], \
                                                int(cur_color[0] * 255), int(cur_color[1] * 255),
                                                int(cur_color[2] * 255)))


def export_ply_with_normal(out, v, n):
    assert v.shape[0] == n.shape[0], 'v.shape[0] != v.shape[0]'

    with open(out, 'w') as fout:
        fout.write('ply\n');
        fout.write('format ascii 1.0\n');
        fout.write('element vertex ' + str(v.shape[0]) + '\n');
        fout.write('property float x\n');
        fout.write('property float y\n');
        fout.write('property float z\n');
        fout.write('property float nx\n');
        fout.write('property float ny\n');
        fout.write('property float nz\n');
        fout.write('end_header\n');

        for i in range(v.shape[0]):
            fout.write('%f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], n[i, 0], n[i, 1], n[i, 2]))


def sample_points_from_obj(label_fn, obj_fn, pts_fn, num_points, verbose=False):
    cmd = 'MeshSample -n%d -s3 -l %s %s %s> /dev/null' % (num_points, label_fn, obj_fn, pts_fn)
    if verbose: print(cmd)
    call(cmd, shell=True)

    with open(pts_fn, 'r') as fin:
        lines = [line.rstrip() for line in fin]
        pts = np.array([[line.split()[0], line.split()[1], line.split()[2]] for line in lines], dtype=np.float32)
        label = np.array([int(line.split()[-1].split('"')[1]) for line in lines], dtype=np.int32)
        if verbose: print('get pts: ', pts.shape)

    return pts, label


def sample_points(v, f, label=None, num_points=200, verbose=False):
    tmp_obj = str(time.time()).replace('.', '_') + '_' + str(random.random()).replace('.', '_') + '.obj'
    tmp_pts = tmp_obj.replace('.obj', '.pts')
    tmp_label = tmp_obj.replace('.obj', '.label')

    if label is None:
        label = np.zeros((f.shape[0]), dtype=np.int32)

    export_obj(tmp_obj, v, f)
    export_label(tmp_label, label)

    pts, fid = sample_points_from_obj(tmp_label, tmp_obj, tmp_pts, num_points=num_points, verbose=verbose)

    cmd = 'rm -rf %s %s %s' % (tmp_obj, tmp_pts, tmp_label)
    call(cmd, shell=True)

    return pts, fid


def export_pts_with_color(out, pc, label):
    num_point = pc.shape[0]
    with open(out, 'w') as fout:
        for i in range(num_point):
            cur_color = label[i]
            fout.write('%f %f %f %d %d %d\n' % (pc[i, 0], pc[i, 1], pc[i, 2], cur_color[0], cur_color[1], cur_color[2]))


def export_pts_with_label(out, pc, label, base=0):
    num_point = pc.shape[0]
    num_colors = len(colors)
    with open(out, 'w') as fout:
        for i in range(num_point):
            cur_color = colors[label[i] % num_colors]
            fout.write('%f %f %f %f %f %f\n' % (pc[i, 0], pc[i, 1], pc[i, 2], cur_color[0], cur_color[1], cur_color[2]))


def export_pts_with_keypoints(out, pc, kp_list):
    num_point = pc.shape[0]
    with open(out, 'w') as fout:
        for i in range(num_point):
            if i in kp_list:
                color = [1.0, 0.0, 0.0]
            else:
                color = [0.0, 0.0, 1.0]

            fout.write('%f %f %f %f %f %f\n' % (pc[i, 0], pc[i, 1], pc[i, 2], color[0], color[1], color[2]))


def compute_boundary_labels(pc, seg, radius=0.05):
    num_points = len(seg)
    assert num_points == pc.shape[0]
    assert pc.shape[1] == 3

    bdr = np.zeros((num_points)).astype(np.int32)

    square_sum = np.sum(pc * pc, axis=1)
    A = np.tile(np.expand_dims(square_sum, axis=0), [num_points, 1])
    B = np.tile(np.expand_dims(square_sum, axis=1), [1, num_points])
    C = np.dot(pc, pc.T)

    dist = A + B - 2 * C

    for i in range(num_points):
        neighbor_seg = seg[dist[i, :] < radius ** 2]
        if len(set(neighbor_seg)) > 1:
            bdr[i] = 1

    return bdr


def render_obj(out, v, f, delete_img=False, flat_shading=True):
    tmp_obj = out.replace('.png', '.obj')

    export_obj(tmp_obj, v, f)

    if flat_shading:
        cmd = 'RenderShape -0 %s %s 600 600 > /dev/null' % (tmp_obj, out)
    else:
        cmd = 'RenderShape %s %s 600 600 > /dev/null' % (tmp_obj, out)

    call(cmd, shell=True)

    img = np.array(imread(out), dtype=np.float32)

    cmd = 'rm -rf %s' % (tmp_obj)
    call(cmd, shell=True)

    if delete_img:
        cmd = 'rm -rf %s' % out
        call(cmd, shell=True)

    return img


def render_obj_with_label(out, v, f, label, delete_img=False, base=0):
    tmp_obj = out.replace('.png', '.obj')
    tmp_label = out.replace('.png', '.label')

    label += base

    export_obj(tmp_obj, v, f)
    export_label(tmp_label, label)

    cmd = 'RenderShape %s -l %s %s 600 600 > /dev/null' % (tmp_obj, tmp_label, out)
    call(cmd, shell=True)

    img = np.array(imread(out), dtype=np.float32)

    cmd = 'rm -rf %s %s' % (tmp_obj, tmp_label)
    call(cmd, shell=True)

    if delete_img:
        cmd = 'rm -rf %s' % out
        call(cmd, shell=True)

    return img


def render_pts_with_label(out, pts, label, delete_img=False, base=0, point_size=6):
    tmp_pts = out.replace('.png', '.pts')
    tmp_label = out.replace('.png', '.label')

    label += base

    export_pts(tmp_pts, pts)
    export_label(tmp_label, label)

    cmd = 'RenderShape %s -l %s %s 600 600 -p %d > /dev/null' % (tmp_pts, tmp_label, out, point_size)
    call(cmd, shell=True)

    img = np.array(imread(out), dtype=np.float32)

    cmd = 'rm -rf %s %s' % (tmp_pts, tmp_label)
    call(cmd, shell=True)

    if delete_img:
        cmd = 'rm -rf %s' % out
        call(cmd, shell=True)

    return img


def render_pts(out, pts, delete_img=False, point_size=6, point_color='FF0000FF'):
    tmp_pts = out.replace('.png', '.pts')
    export_pts(tmp_pts, pts)

    cmd = 'RenderShape %s %s 600 600 -p %d -c %s > /dev/null' % (tmp_pts, out, point_size, point_color)
    call(cmd, shell=True)

    img = np.array(imread(out), dtype=np.float32)

    cmd = 'rm -rf %s' % tmp_pts
    call(cmd, shell=True)

    if delete_img:
        cmd = 'rm -rf %s' % out
        call(cmd, shell=True)

    return img


def render_pts_with_keypoints(out, pts, kp_list, delete_img=False, \
                              point_size=6, fancy_kp=False, fancy_kp_num=20, fancy_kp_radius=0.02):
    tmp_pts = out.replace('.png', '.pts')
    tmp_label = out.replace('.png', '.label')

    num_point = pts.shape[0]
    labels = np.ones((num_point), dtype=np.int32) * 14

    for idx in kp_list:
        labels[idx] = 13

    if fancy_kp:
        num_kp = len(kp_list)
        more_pts = np.zeros((num_kp * fancy_kp_num, 3), dtype=np.float32)
        more_labels = np.ones((num_kp * fancy_kp_num), dtype=np.int32) * 13

        for i, idx in enumerate(kp_list):
            for j in range(fancy_kp_num):
                x = np.random.randn()
                y = np.random.randn()
                z = np.random.randn()

                l = np.sqrt(x ** 2 + y ** 2 + z ** 2)

                x = x / l * fancy_kp_radius + pts[idx, 0]
                y = y / l * fancy_kp_radius + pts[idx, 1]
                z = z / l * fancy_kp_radius + pts[idx, 2]

                more_pts[i * fancy_kp_num + j, 0] = x
                more_pts[i * fancy_kp_num + j, 1] = y
                more_pts[i * fancy_kp_num + j, 2] = z

        pts = np.concatenate((pts, more_pts), axis=0)
        labels = np.concatenate((labels, more_labels), axis=0)

    export_pts(tmp_pts, pts)
    export_label(tmp_label, labels)

    cmd = 'RenderShape %s -l %s %s 600 600 -p %d > /dev/null' % (tmp_pts, tmp_label, out, point_size)
    call(cmd, shell=True)

    img = np.array(imread(out), dtype=np.float32)

    cmd = 'rm -rf %s %s' % (tmp_pts, tmp_label)
    call(cmd, shell=True)

    if delete_img:
        cmd = 'rm -rf %s' % out
        call(cmd, shell=True)

    return img


def compute_normal(pts, neighbor=50):
    l = pts.shape[0]
    assert (l > neighbor)
    t = np.sum(pts ** 2, axis=1)
    A = np.tile(t, (l, 1))
    C = np.array(A).T
    B = np.dot(pts, pts.T)
    dist = A - 2 * B + C

    neigh_ids = dist.argsort(axis=1)[:, :neighbor]
    vec_ones = np.ones((neighbor, 1)).astype(np.float32)
    normals = np.zeros((l, 3)).astype(np.float32)
    for idx in range(l):
        D = pts[neigh_ids[idx, :], :]
        cur_normal = np.dot(np.linalg.pinv(D), vec_ones)
        cur_normal = np.squeeze(cur_normal)
        len_normal = np.sqrt(np.sum(cur_normal ** 2))
        normals[idx, :] = cur_normal / len_normal

        if np.dot(normals[idx, :], pts[idx, :]) < 0:
            normals[idx, :] = -normals[idx, :]

    return normals


def transfer_label_from_pts_to_obj(vertices, faces, pts, label):
    assert pts.shape[0] == label.shape[0], 'ERROR: #pts != #label'
    num_pts = pts.shape[0]

    num_faces = faces.shape[0]
    face_centers = []
    for i in range(num_faces):
        face_centers.append(
            (vertices[faces[i, 0] - 1, :] + vertices[faces[i, 1] - 1, :] + vertices[faces[i, 2] - 1, :]) / 3)
    face_center_array = np.vstack(face_centers)

    A = np.tile(np.expand_dims(np.sum(face_center_array ** 2, axis=1), axis=0), [num_pts, 1])
    B = np.tile(np.expand_dims(np.sum(pts ** 2, axis=1), axis=1), [1, num_faces])
    C = np.dot(pts, face_center_array.T)
    dist = A + B - 2 * C

    lid = np.argmax(-dist, axis=0)
    face_label = label[lid]
    return face_label


def detect_connected_component(vertices, faces, face_labels=None):
    edge2facelist = dict()

    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]

    bar = progressbar.ProgressBar()
    face_id_list = []
    for face_id in bar(range(num_faces)):
        f0 = faces[face_id, 0] - 1
        f1 = faces[face_id, 1] - 1
        f2 = faces[face_id, 2] - 1
        id_list = np.sort([f0, f1, f2])
        s0 = id_list[0]
        s1 = id_list[1]
        s2 = id_list[2]

        key1 = '%d_%d' % (s0, s1)
        if key1 in edge2facelist.keys():
            edge2facelist[key1].append(face_id)
        else:
            edge2facelist[key1] = [face_id]

        key2 = '%d_%d' % (s1, s2)
        if key2 in edge2facelist.keys():
            edge2facelist[key2].append(face_id)
        else:
            edge2facelist[key2] = [face_id]

        key3 = '%d_%d' % (s0, s2)
        if key3 in edge2facelist.keys():
            edge2facelist[key3].append(face_id)
        else:
            edge2facelist[key3] = [face_id]

        face_id_list.append([key1, key2, key3])

    face_used = np.zeros((num_faces), dtype=np.bool);
    face_seg_id = np.zeros((num_faces), dtype=np.int32);
    cur_id = 0;

    new_part = False
    for i in range(num_faces):
        q = deque()
        q.append(i)
        while len(q) > 0:
            face_id = q.popleft()
            if not face_used[face_id]:
                face_used[face_id] = True
                new_part = True
                face_seg_id[face_id] = cur_id
                for key in face_id_list[face_id]:
                    for new_face_id in edge2facelist[key]:
                        if not face_used[new_face_id] and (face_labels == None or
                                                           face_labels[new_face_id] == face_labels[face_id]):
                            q.append(new_face_id)

        if new_part:
            cur_id += 1
            new_part = False

    return face_seg_id


def calculate_two_pts_distance(pts1, pts2):
    A = np.tile(np.expand_dims(np.sum(pts1 ** 2, axis=1), axis=-1), [1, pts2.shape[0]])
    B = np.tile(np.expand_dims(np.sum(pts2 ** 2, axis=1), axis=0), [pts1.shape[0], 1])
    C = np.dot(pts1, pts2.T)
    dist = A + B - 2 * C
    return dist


def propagate_pts_seg_from_another_pts(ori_pts, ori_seg, tar_pts):
    dist = calculate_two_pts_distance(ori_pts, tar_pts)
    idx = np.argmin(dist, axis=0)
    return ori_seg[idx]


# ----------------------------------------
# Testing
# ----------------------------------------
if __name__ == '__main__':
    print('running some tests')

    ############
    ## Test "write_lines_as_cylinders"
    ############
    pcl = np.random.rand(32, 2, 3)
    write_lines_as_cylinders(pcl, 'point_connectors')
    input()

    scene_bbox = np.zeros((1, 7))
    scene_bbox[0, 3:6] = np.array([1, 2, 3])  # dx,dy,dz
    scene_bbox[0, 6] = np.pi / 4  # 45 degrees
    write_oriented_bbox(scene_bbox, 'single_obb_45degree.ply')
    ############
    ## Test point_cloud_to_bbox
    ############
    pcl = np.random.rand(32, 16, 3)
    pcl_bbox = point_cloud_to_bbox(pcl)
    assert pcl_bbox.shape == (32, 6)

    pcl = np.random.rand(16, 3)
    pcl_bbox = point_cloud_to_bbox(pcl)
    assert pcl_bbox.shape == (6,)

    ############
    ## Test corner distance
    ############
    crnr1 = np.array([[2.59038660e+00, 8.96107932e-01, 4.73305349e+00],
                      [4.12281644e-01, 8.96107932e-01, 4.48046631e+00],
                      [2.97129656e-01, 8.96107932e-01, 5.47344275e+00],
                      [2.47523462e+00, 8.96107932e-01, 5.72602993e+00],
                      [2.59038660e+00, 4.41155793e-03, 4.73305349e+00],
                      [4.12281644e-01, 4.41155793e-03, 4.48046631e+00],
                      [2.97129656e-01, 4.41155793e-03, 5.47344275e+00],
                      [2.47523462e+00, 4.41155793e-03, 5.72602993e+00]])
    crnr2 = crnr1

    print(bbox_corner_dist_measure(crnr1, crnr2))

    print('tests PASSED')

