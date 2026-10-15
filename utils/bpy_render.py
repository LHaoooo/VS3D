import json
import math
import os
from typing import Dict, Tuple

import bpy
import numpy as np
from mathutils import Vector


class BpyRenderer:

    def __init__(
        self,
        resolution: int = 512,
        engine: str = "CYCLES",
        geo_mode: bool = False,
        split_normal: bool = False,
    ):
        self.resolution = resolution
        self.engine = engine
        self.geo_mode = geo_mode
        self.split_normal = split_normal
        self.import_functions = self._setup_import_functions()

    def _setup_import_functions(self):
        import_functions = {
            "obj": bpy.ops.wm.obj_import,
            "glb": bpy.ops.import_scene.gltf,
            "gltf": bpy.ops.import_scene.gltf,
            "usd": bpy.ops.import_scene.usd,
            "fbx": bpy.ops.import_scene.fbx,
            "stl": bpy.ops.import_mesh.stl,
            "usda": bpy.ops.import_scene.usda,
            "dae": bpy.ops.wm.collada_import,
            "ply": bpy.ops.wm.ply_import,
            "abc": bpy.ops.wm.alembic_import,
            "blend": bpy.ops.wm.append,
        }

        return import_functions

    def init_render_settings(self):
        '''
        设置基础渲染参数 → 分辨率、透明背景、输出格式。
        配置 Cycles 渲染器 → 根据 geo_mode 决定是快速预览还是高质量渲染。
        启用 GPU 加速 → CUDA 优先，加快渲染速度。
        '''
        bpy.context.scene.render.engine = self.engine  # "CYCLES" or "BLENDER_EEVEE"
        bpy.context.scene.render.resolution_x = self.resolution  # 512
        bpy.context.scene.render.resolution_y = self.resolution  # 512
        bpy.context.scene.render.resolution_percentage = 100  # 完整分辨率
        bpy.context.scene.render.image_settings.file_format = "PNG" # 设置输出图像格式为 PNG，带透明通道 (RGBA)
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.film_transparent = True  # 渲染结果背景透明

        if self.engine == "CYCLES":
            bpy.context.scene.render.engine = "CYCLES"
            bpy.context.scene.cycles.samples = 128 if not self.geo_mode else 1  # 采样数128（高质量），越高噪点越少，渲染时间越长 如果是 geo_mode 就用 1（快速预览）
            bpy.context.scene.cycles.filter_type = "BOX"
            bpy.context.scene.cycles.filter_width = 1  # 使用 BOX 滤波，宽度 1 像素（最简单最快）
            bpy.context.scene.cycles.diffuse_bounces = 1  # 漫反射只允许 1 次
            bpy.context.scene.cycles.glossy_bounces = 1  # 光泽（镜面）反射只允许 1 次
            bpy.context.scene.cycles.transparent_max_bounces = (
                3 if not self.geo_mode else 0
            )  # 透明材质最多 3 次 (geo_mode 下禁用)
            bpy.context.scene.cycles.transmission_bounces = (
                3 if not self.geo_mode else 1
            )  # 折射光线最多 3 次 (geo_mode 下仅 1 次)
            bpy.context.scene.cycles.use_denoising = True  # 开启去噪（减少渲染噪点）

            # 指定设备加速设置
            try:
                bpy.context.scene.cycles.device = "GPU"
                bpy.context.preferences.addons["cycles"].preferences.get_devices()
                bpy.context.preferences.addons[
                    "cycles"
                ].preferences.compute_device_type = "CUDA"
            except:
                pass

    def init_scene(self):
        # 遍历所有的对象、材质、贴图、图像并全部删除
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)

        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture, do_unlink=True)

        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)

    def init_camera(self):
        # 在场景里新建一台相机，并设置好“始终看向场景中心(0,0,0)”
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam  # 设置为主相机
        cam.data.sensor_height = cam.data.sensor_width = 32  # 传感器尺寸 32mm

        cam_constraint = cam.constraints.new(type="TRACK_TO")  # 相机约束：始终看向场景中心
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"  # 相机的 -Z 轴指向目标
        cam_constraint.up_axis = "UP_Y"  # 相机的 Y 轴向上

        # 相机会自动旋转，让它的视野一直对准原点
        cam_empty = bpy.data.objects.new("Empty", None)  # 新建一个空对象作为相机的目标
        cam_empty.location = (0, 0, 0)  # 目标位置在场景中心
        bpy.context.scene.collection.objects.link(cam_empty)
        cam_constraint.target = cam_empty

        return cam  # 返回相机对象

    def init_lighting(self):
        '''初始化场景光照 → 删除默认光源，添加新的点光源和面光源。'''
        # 清空场景里原本的灯光
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()
        # 新建点光源作为基础光源，提供照明和方向感
        default_light = bpy.data.objects.new(
            "Default_Light", bpy.data.lights.new("Default_Light", type="POINT")
        )
        bpy.context.collection.objects.link(default_light)
        default_light.data.energy = 1000
        default_light.location = (4, 1, 6)
        default_light.rotation_euler = (0, 0, 0)
        # 新建面光源作为顶光
        top_light = bpy.data.objects.new(
            "Top_Light", bpy.data.lights.new("Top_Light", type="AREA")
        )
        bpy.context.collection.objects.link(top_light)
        top_light.data.energy = 10000
        top_light.location = (0, 0, 10)
        top_light.scale = (100, 100, 100)
        # 再建一个面光源作为底光
        bottom_light = bpy.data.objects.new(
            "Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA")
        )
        bpy.context.collection.objects.link(bottom_light)
        bottom_light.data.energy = 1000
        bottom_light.location = (0, 0, -10)
        bottom_light.rotation_euler = (0, 0, 0)

        return {
            "default_light": default_light,
            "top_light": top_light,
            "bottom_light": bottom_light,
        }  # 返回三种光源对象

    def load_object(self, object_path: str):
        file_extension = object_path.split(".")[-1].lower()

        if file_extension not in self.import_functions:
            raise ValueError(f"Unsupported file type: {file_extension}")

        import_function = self.import_functions[file_extension]

        print(f"Loading object from {object_path}")

        if file_extension == "blend":
            import_function(directory=object_path, link=False)
        elif file_extension in {"glb", "gltf"}:
            import_function(
                filepath=object_path, merge_vertices=True, import_shading="NORMALS"
            )  # 合并重复顶点,使用法线作为 shading 信息
        else:
            import_function(filepath=object_path)

    def delete_invisible_objects(self):
        '''删除场景中隐藏的对象和集合'''
        bpy.ops.object.select_all(action="DESELECT")  # 取消选择所有对象
        # 删除隐藏的对象
        for obj in bpy.context.scene.objects:
            if obj.hide_viewport or obj.hide_render:
                obj.hide_viewport = False
                obj.hide_render = False
                obj.hide_select = False
                obj.select_set(True)
        bpy.ops.object.delete()

        invisible_collections = [
            col for col in bpy.data.collections if col.hide_viewport
        ]
        for col in invisible_collections:
            bpy.data.collections.remove(col)

    def unhide_all_objects(self):
        '''取消场景中所有对象的隐藏状态'''
        for obj in bpy.context.scene.objects:
            obj.hide_set(False)

    def convert_to_meshes(self):
        # 把场景中的所有对象（无论类型是什么）都批量转换成 Mesh 网格对象
        bpy.ops.object.select_all(action="DESELECT")  # 取消选择所有对象
        bpy.context.view_layer.objects.active = [
            obj for obj in bpy.context.scene.objects if obj.type == "MESH"
        ][0]  # 选中第一个mesh对象作为活动对象
        for obj in bpy.context.scene.objects:
            obj.select_set(True)  # 选择所有对象
        bpy.ops.object.convert(target="MESH")  # 把选中的对象全部转换成 Mesh 类型

    def triangulate_meshes(self):
        bpy.ops.object.select_all(action="DESELECT")  # 取消选择场景中所有对象
        objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]  # 获取场景中所有的 Mesh 对象
        bpy.context.view_layer.objects.active = objs[0]  # 选中第一个mesh对象作为活动对象
        for obj in objs:
            obj.select_set(True)  # 选择所有 Mesh 对象
        bpy.ops.object.mode_set(mode="EDIT")  # 进入编辑模式
        bpy.ops.mesh.reveal()  # 显示所有隐藏的几何体
        bpy.ops.mesh.select_all(action="SELECT")  # 选择所有几何体
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")  # 将四边形和多边形（ngon）转换为三角形
        bpy.ops.object.mode_set(mode="OBJECT")  # 切回对象模式
        bpy.ops.object.select_all(action="DESELECT")  # 取消选择所有对象

    def split_mesh_normal(self):
        bpy.ops.object.select_all(action="DESELECT")
        objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        bpy.context.view_layer.objects.active = objs[0]
        for obj in objs:
            obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.split_normals()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

    def delete_custom_normals(self):
        for this_obj in bpy.data.objects:
            if this_obj.type == "MESH":
                bpy.context.view_layer.objects.active = this_obj
                bpy.ops.mesh.customdata_custom_splitnormals_clear()

    def override_material(self):
        new_mat = bpy.data.materials.new(name="Override0123456789")
        new_mat.use_nodes = True
        new_mat.node_tree.nodes.clear()
        bsdf = new_mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
        bsdf.inputs[1].default_value = 1
        output = new_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        new_mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        bpy.context.scene.view_layers["View Layer"].material_override = new_mat

    def scene_bbox(self) -> Tuple[Vector, Vector]:
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False

        scene_meshes = [
            obj
            for obj in bpy.context.scene.objects.values()
            if isinstance(obj.data, bpy.types.Mesh)
        ]
        for obj in scene_meshes:
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")

        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene(self) -> Tuple[float, Vector]:
        '''把整个 3D 场景的大小和位置归一化，让它居中在原点，并且缩放到标准大小。'''
        scene_root_objects = [
            obj for obj in bpy.context.scene.objects.values() if not obj.parent
        ]  # 获取所有根对象（没有父对象的对象）

        # 如果场景里有多个根物体，就新建一个空对象 "ParentEmpty"，把所有根物体的 parent 设为它。→ 这样可以当作一个“整体”来缩放和平移。
        # 如果只有一个物体，就直接把它当作根。
        if len(scene_root_objects) > 1:
            scene = bpy.data.objects.new("ParentEmpty", None)
            bpy.context.scene.collection.objects.link(scene)

            for obj in scene_root_objects:
                obj.parent = scene
        else:
            scene = scene_root_objects[0]
        # 计算整个场景的bbox
        bbox_min, bbox_max = self.scene_bbox()
        print(f"[INFO] Bounding box: {bbox_min}, {bbox_max}")
        scale = 1 / max(bbox_max - bbox_min)  # 计算场景的缩放系数：取包围盒最大边长的倒数。这样缩放之后，场景的最大尺寸就变成1
        scene.scale = scene.scale * scale

        bpy.context.view_layer.update()
        bbox_min, bbox_max = self.scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        scene.matrix_world.translation += offset  # 把整个场景平移，使其居中
        bpy.ops.object.select_all(action="DESELECT")  # 取消所有选中

        return scale, offset  # 返回缩放系数和偏移量


    def glb_to_ply(
        self,
        glb_path: str,
        ply_path: str,
        scale: float = 1.0,
        offset: Vector = None,
    ) -> Dict:
        """
        Only export mesh to PLY (e.g., GLB -> PLY).
        """
        output_dir = os.path.dirname(os.path.abspath(ply_path)) or "."
        os.makedirs(output_dir, exist_ok=True)

        # 1) Load scene/object
        self.init_scene()
        self.load_object(glb_path)
        if getattr(self, "split_normal", False):
            self.split_mesh_normal()
        print("[INFO] Scene initialized for mesh export.")

        # 2) Normalize scene
        if offset is None:
            scale, offset = self.normalize_scene()
            print(f"[INFO] Scene normalized with auto scale: {scale}, offset: {offset}")
        else:
            scene_root_objects = [
                obj for obj in bpy.context.scene.objects.values() if not obj.parent
            ]
            if len(scene_root_objects) > 1:
                scene = bpy.data.objects.new("ParentEmpty", None)
                bpy.context.scene.collection.objects.link(scene)
                for obj in scene_root_objects:
                    obj.parent = scene
            else:
                scene = scene_root_objects[0]

            scene.scale = scene.scale * scale
            bpy.context.view_layer.update()
            scene.matrix_world.translation += offset
            bpy.ops.object.select_all(action="DESELECT")

            print(f"[INFO] Scene scaled with specified scale: {scale}, offset: {offset}")

        # 3) Export mesh as PLY
        mesh_file_path = None
        try:
            self.unhide_all_objects()
            self.convert_to_meshes()
            self.triangulate_meshes()
            print("[INFO] Meshes triangulated.")

            try:
                bpy.ops.wm.ply_export(filepath=ply_path)
                mesh_file_path = ply_path
                print("[INFO] Mesh file saved via bpy.ops.wm.ply_export.")
            except AttributeError:
                try:
                    bpy.ops.export_mesh.ply(filepath=ply_path)
                    mesh_file_path = ply_path
                    print("[INFO] Mesh file saved via bpy.ops.export_mesh.ply.")
                except AttributeError:
                    print("[WARNING] PLY export op not available, skipping mesh export")
        except Exception as e:
            print(f"[WARNING] GLB->PLY export failed: {e}")

        return {
            "exported": mesh_file_path is not None,
            "glb_path": glb_path,
            "ply_path": mesh_file_path,
            "scale": scale,
            "offset": [offset.x, offset.y, offset.z] if offset is not None else None,
        }
