"""
Author: Luigi Piccinelli
Licensed under the CC BY-NC-SA 4.0 license (http://creativecommons.org/licenses/by-nc-sa/4.0/)
"""

import gc
import os
import shutil
import time
from datetime import datetime
from math import pi

import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image

from unik3d.models import UniK3D
from unik3d.utils.camera import OPENCV, Fisheye624, Pinhole, Spherical


def predictions_to_glb(
    predictions,
    mask_black_bg=False,
    mask_far_points=False,
) -> trimesh.Scene:
    print("Building GLB scene")
    images = predictions["image"].squeeze().permute(1, 2, 0).cpu().numpy()
    world_points = predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy()

    vertices_3d = world_points.reshape(-1, 3)
    # flip x and y
    vertices_3d[:, 1] *= -1
    vertices_3d[:, 0] *= -1
    colors_rgb = (images.reshape(-1, 3)).astype(np.uint8)

    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        vertices_3d = vertices_3d[black_bg_mask]
        colors_rgb = colors_rgb[black_bg_mask]

    if mask_far_points:
        far_points_mask = np.linalg.norm(vertices_3d, axis=-1) < 100.0
        vertices_3d = vertices_3d[far_points_mask]
        colors_rgb = colors_rgb[far_points_mask]

    scene_3d = trimesh.Scene()
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud_data)

    return scene_3d


def instantiate_model(model_name):
    type_ = model_name[0].lower()

    name = f"unik3d-vit{type_}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")

    # Set resolution level and interpolation mode as specified.
    model.resolution_level = 9
    model.interpolation_mode = "bilinear"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model


def instantiate_camera(camera_name, params, device):
    if camera_name == "Predicted":
        return None
    fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, t1, t2, hfov, H, W = params
    if camera_name == "Pinhole":
        params = [fx, fy, cx, cy]
    elif camera_name == "Fisheye624":
        params = [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, t1, t2]
    elif camera_name == "OPENCV":
        params = [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, t1, t2]
    elif camera_name == "Equirectangular":
        # dummy intrinsics for spherical camera, assume hfov -> vfov based on input shapes
        hfov2 = hfov * pi / 180.0 / 2
        params = [fx, fy, cx, cy, W, H, hfov2, H / W * hfov2]
        camera_name = "Spherical"

    return eval(camera_name)(params=torch.tensor(params).float()).to(device)


def run_model(target_dir, model_name, camera_name, params, efficiency):

    print("Instantiating model and camera...")
    model = instantiate_model(model_name)

    image_names = [x for x in os.listdir(target_dir) if x.endswith(".png")]
    input_image = np.array(Image.open(os.path.join(target_dir, image_names[-1])))
    image_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    H, W = image_tensor.shape[-2:]
    params = params + [H, W]
    camera = instantiate_camera(camera_name, params=params, device=device)

    # Perform inference with the model.
    print("Running inference...")
    model.resolution_level = min(efficiency, 9.0)
    outputs = model.infer(image_tensor, camera=camera, normalize=True)
    outputs["image"] = image_tensor

    return outputs


def gradio_demo(
    target_dir,
    model_name,
    camera_name,
    fx,
    fy,
    cx,
    cy,
    k1,
    k2,
    k3,
    k4,
    k5,
    k6,
    t1,
    t2,
    hfov,
    mask_black_bg,
    mask_far_points,
    efficiency
):
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None

    start_time = time.time()
    gc.collect()

    print("Running run_model...")
    params = [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, t1, t2, hfov]
    with torch.no_grad():
        outputs = run_model(target_dir, model_name, camera_name, params, efficiency)

    # Save predictions
    points = outputs["points"].squeeze().permute(1, 2, 0).cpu().numpy()
    rgb = outputs["image"].squeeze().permute(1, 2, 0).cpu().numpy()

    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, {"points": points, "image": rgb})

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        outputs,
        mask_black_bg=mask_black_bg,
        mask_far_points=mask_far_points,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del outputs
    gc.collect()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    log_msg = f"Success. Waiting for visualization."

    return glbfile, log_msg, prediction_save_path


def handle_uploads(input_image):
    gc.collect()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    target_dir = os.path.join(tmpdir, f"input_images_{timestamp}")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    dst_path = os.path.join(target_dir, "image.png")
    Image.fromarray(input_image).save(dst_path)
    image_paths = [dst_path]

    print(f"Files uploaded.")
    return target_dir, image_paths


def update_gallery_on_upload(input_images):
    if input_images is None:
        return None, None
    target_dir, image_path = handle_uploads(input_images)
    return target_dir, "Upload complete. Click 'Run UniK3D' to get 3D pointcloud."


def update_parameters(camera):
    if camera == "Pinhole":
        return (
            gr.update(visible=True),  # fx
            gr.update(visible=True),  # fy
            gr.update(visible=True),  # cx
            gr.update(visible=True),  # cy
            gr.update(visible=False),  # k1
            gr.update(visible=False),  # k2
            gr.update(visible=False),  # k3
            gr.update(visible=False),  # k4
            gr.update(visible=False),  # k5
            gr.update(visible=False),  # k6
            gr.update(visible=False),  # t1
            gr.update(visible=False),  # t2
            gr.update(visible=False),  # hfov
        )
    elif camera == "OPENCV":
        return (
            gr.update(visible=True),  # fx
            gr.update(visible=True),  # fy
            gr.update(visible=True),  # cx
            gr.update(visible=True),  # cy
            gr.update(visible=True),  # k1
            gr.update(visible=True),  # k2
            gr.update(visible=True),  # k3
            gr.update(visible=False),  # k4
            gr.update(visible=False),  # k5
            gr.update(visible=False),  # k6
            gr.update(visible=True),  # t1
            gr.update(visible=True),  # t2
            gr.update(visible=False),  # hfov
        )
    elif camera == "Fisheye624":
        return (
            gr.update(visible=True),  # fx
            gr.update(visible=True),  # fy
            gr.update(visible=True),  # cx
            gr.update(visible=True),  # cy
            gr.update(visible=True),  # k1
            gr.update(visible=True),  # k2
            gr.update(visible=True),  # k3
            gr.update(visible=True),  # k4
            gr.update(visible=True),  # k5
            gr.update(visible=True),  # k6
            gr.update(visible=True),  # t1
            gr.update(visible=True),  # t2
            gr.update(visible=False),  # hfov
        )
    elif camera == "Equirectangular":
        return (
            gr.update(visible=False),  # fx
            gr.update(visible=False),  # fy
            gr.update(visible=False),  # cx
            gr.update(visible=False),  # cy
            gr.update(visible=False),  # k1
            gr.update(visible=False),  # k2
            gr.update(visible=False),  # k3
            gr.update(visible=False),  # k4
            gr.update(visible=False),  # k5
            gr.update(visible=False),  # k6
            gr.update(visible=False),  # t1
            gr.update(visible=False),  # t2
            gr.update(visible=True),  # hfov
        )
    elif camera == "Predicted":
        return (
            gr.update(visible=False),  # fx
            gr.update(visible=False),  # fy
            gr.update(visible=False),  # cx
            gr.update(visible=False),  # cy
            gr.update(visible=False),  # k1
            gr.update(visible=False),  # k2
            gr.update(visible=False),  # k3
            gr.update(visible=False),  # k4
            gr.update(visible=False),  # k5
            gr.update(visible=False),  # k6
            gr.update(visible=False),  # t1
            gr.update(visible=False),  # t2
            gr.update(visible=False),  # hfov
        )
    else:
        raise ValueError(f"Invalid camera type: {camera}")


def clear_fields():
    return None


def update_log():
    return "Loading Model and Running Inference..."


def update_visualization(target_dir, mask_black_bg, mask_far_points, is_example):

    if is_example == "True":
        return (
            None,
            "No reconstruction available. Please click the Reconstruct button first.",
        )

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return (
            None,
            "No reconstruction available. Please click the Reconstruct button first.",
        )

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return (
            None,
            f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first.",
        )

    loaded = np.load(predictions_path, allow_pickle=True)
    predictions = {key: loaded[key] for key in loaded.keys()}

    glbfile = os.path.join(
        target_dir,
        f"glbscene.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            mask_black_bg=mask_black_bg,
            mask_far_points=mask_far_points,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


if __name__ == "__main__":
    theme = gr.themes.Citrus()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    with gr.Blocks(
        theme=theme,
        css="""
        .custom-log * {
            font-style: italic;
            font-size: 22px !important;
            background-image: linear-gradient(120deg, #ff7e26 0%, #ff9c59 60%, #fff4d6 100%);
            -webkit-background-clip: text;
            background-clip: text;
            font-weight: bold !important;
            color: transparent !important;
            text-align: center !important;
        }
        
        .example-log * {
            font-style: italic;
            font-size: 16px !important;
            background-image: linear-gradient(120deg, #ff7e26 0%, #ff9c59 60%, #fff4d6 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent !important;
        }
        
        #my_radio .wrap {
            display: flex;
            flex-wrap: nowrap;
            justify-content: center;
            align-items: center;
        }

        #my_radio .wrap label {
            display: flex;
            width: 50%;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 10px 0;
            box-sizing: border-box;
        }
        """,
    ) as demo:

        # Instead of gr.State, we use a hidden Textbox:
        is_example = gr.Textbox(label="is_example", visible=False, value="None")

        gr.HTML(
            """
        <h1>UniK3D: Universal Camera Monocular 3D Estimation</h1>
        <p>
        <a href="https://github.com/lpiccinelli-eth/UniK3D">🌟 GitHub Repository</a> |
        <a href="">🚀 Project Page</a>
        </p>

        <div style="font-size: 16px; line-height: 1.5;">
        <p>Upload one image to create a 3D estimation of a scene or object. UniK3D allows to predict directly 3D of any camera and scene.</p>

        <h3>Getting Started:</h3>
        <ol>
            <li><strong>Upload Your Image:</strong> Use the "Upload Images" panel to provide your input.</li>
            <li><strong>Run:</strong> Click the "Run UniK3D" button to start the 3D estimation process.</li>
            <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file.</li>
            <li><strong>Downstream:</strong> The 3D output can be used as reconstruction or for monocular camera calibration.</li>
        </ol>
        <p><strong style="color: #ff7e26;">Please note:</strong> <span style="color: #ff7e26; font-weight: bold;">Our model runs on CPU on HuggingFace Space. Actual inference is less than 100ms second per image on consumer-level GPUs, on Spaces will take between 20s and 90s, depending on the "Speed-Resoltion Tradeoff" chosen. Web-based 3D pointcloud visualization may be slow due to Gradio's rendering. For faster visualization, use a local machine to run our demo from our <a href="https://github.com/lpiccinelli-eth/UniK3D">GitHub repository</a>. </span></p>
        </div>
        """
        )

        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

        with gr.Row():
            with gr.Column():
                camera_model = gr.Dropdown(
                    choices=[
                        "Predicted",
                        "Pinhole",
                        "Fisheye624",
                        "OPENCV",
                        "Equirectangular",
                    ],
                    label="Input Camera",
                )
                model_size = gr.Dropdown(
                    choices=["Large", "Base", "Small"], label="Utilized Model"
                )
                mask_black_bg = gr.Checkbox(
                    label="Filter Black Background", value=False
                )
                mask_far_points = gr.Checkbox(label="Filter Far Points", value=False)
                efficiency = gr.Slider(0, 10, step=1, value=10, label="Speed-Resolution Tradeoff", info="Lower is faster and Higher is more detailed")

            with gr.Column():
                fx = gr.Number(label="Focal length x", value=500.0, visible=False)
                fy = gr.Number(label="Focal length y", value=500.0, visible=False)
                cx = gr.Number(label="Center projection x", value=320.0, visible=False)
                cy = gr.Number(label="Center projection y", value=240.0, visible=False)
                hfov = gr.Number(
                    label="Horizontal FoV (degree)", value=0.0, visible=False
                )

            with gr.Column():
                k1 = gr.Number(label="Radial 1", value=0.0, visible=False)
                k2 = gr.Number(label="Radial 2", value=0.0, visible=False)
                k3 = gr.Number(label="Radial 3", value=0.0, visible=False)
                k4 = gr.Number(label="Radial 4", value=0.0, visible=False)

            with gr.Column():
                k5 = gr.Number(label="Radial 5", value=0.0, visible=False)
                k6 = gr.Number(label="Radial 6", value=0.0, visible=False)
                t1 = gr.Number(label="Tangential 1", value=0.0, visible=False)
                t2 = gr.Number(label="Tangential 2", value=0.0, visible=False)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Images")
                gr.Markdown("**3D Estimation**")
                with gr.Row():
                    log_output = gr.Markdown(
                        "Please upload one image at a time, then click `Run UniK3D`.",
                        elem_classes=["custom-log"],
                    )
                    reconstruction_npy = gr.File(
                        label="Download 3D Pointcloud", type="filepath"
                    )

            with gr.Column(scale=2):
                reconstruction_output = gr.Model3D(
                    height=520, zoom_speed=0.5, pan_speed=0.5
                )
                with gr.Row():
                    submit_btn = gr.Button("Run UniK3D", scale=1, variant="primary")
                    clear_btn = gr.ClearButton(
                        [
                            input_image,
                            reconstruction_output,
                            log_output,
                            target_dir_output,
                            reconstruction_npy,
                        ],
                        scale=1,
                    )

        examples = [
            [
                "assets/demo/poorthings.jpg",
                "Large",
                "Predicted",
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                True,
                False,
                10.0,
            ],
            [
                "assets/demo/naruto.jpg",
                "Large",
                "Predicted",
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                False,
                False,
                10.0,
            ],
            [
                "assets/demo/bears.jpg",
                "Large",
                "Predicted",
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                True,
                False,
                10.0,
            ],
            [
                "assets/demo/berzirk.jpg",
                "Large",
                "Predicted",
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                True,
                False,
                10.0,
            ],
            [
                "assets/demo/luke.webp",
                "Large",
                "Predicted",
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                False,
                False,
                10.0,
            ],
            [
                "assets/demo/equirectangular.jpg",
                "Large",
                "Equirectangular",
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                360.0,
                False,
                False,
                10.0,
            ],
            [
                "assets/demo/venice.jpg",
                "Large",
                "Equirectangular",
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                360.0,
                False,
                True,
                10.0,
            ],
            [
                "assets/demo/dl3dv.png",
                "Large",
                "OPENCV",
                429.57611083984375,
                429.6898193359375,
                479.5,
                269.5,
                -0.0014844092074781656,
                0.0007422995404340327,
                0.0,
                0.0,
                0.0,
                0.0,
                0.00012013866944471374,
                0.001125041046179831,
                0.0,
                False,
                False,
                10.0,
            ],
            [
                "assets/demo/scannet.jpg",
                "Large",
                "Fisheye624",
                791.90869140625,
                792.7230834960938,
                878.16796875,
                585.045166015625,
                -0.029167557135224342,
                -0.006803446915000677,
                -0.0012682401575148106,
                -4.6094228309812024e-05,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                False,
                False,
                10.0,
            ],
        ]

        def example_pipeline(
            input_image,
            model_name,
            camera_name,
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            t1,
            t2,
            hfov,
            mask_black_bg,
            mask_far_points,
            efficiency
        ):
            target_dir, image_path = handle_uploads(input_image)
            glbfile, log_msg, prediction_save_path = gradio_demo(
                target_dir,
                model_name,
                camera_name,
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
                k5,
                k6,
                t1,
                t2,
                hfov,
                mask_black_bg,
                mask_far_points,
                efficiency
            )
            return (
                glbfile,
                log_msg,
                prediction_save_path,
                target_dir,
                image_path,
            )

        gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

        gr.Examples(
            examples=examples,
            inputs=[
                input_image,
                model_size,
                camera_model,
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
                k5,
                k6,
                t1,
                t2,
                hfov,
                mask_black_bg,
                mask_far_points,
                efficiency
            ],
            outputs=[reconstruction_output, log_output, reconstruction_npy],
            fn=example_pipeline,
            cache_examples=False,
            examples_per_page=50,
        )

        submit_btn.click(
            fn=clear_fields, inputs=[], outputs=[reconstruction_output]
        ).then(fn=update_log, inputs=[], outputs=[log_output]).then(
            fn=gradio_demo,
            inputs=[
                target_dir_output,
                model_size,
                camera_model,
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
                k5,
                k6,
                t1,
                t2,
                hfov,
                mask_black_bg,
                mask_far_points,
                efficiency
            ],
            outputs=[reconstruction_output, log_output, reconstruction_npy],
        ).then(
            fn=lambda: "False", inputs=[], outputs=[is_example]
        )

        mask_black_bg.change(
            update_visualization,
            [target_dir_output, mask_black_bg, mask_far_points, is_example],
            [reconstruction_output, log_output],
        )

        mask_far_points.change(
            update_visualization,
            [target_dir_output, mask_black_bg, mask_far_points, is_example],
            [reconstruction_output, log_output],
        )

        input_image.change(
            fn=update_gallery_on_upload,
            inputs=[input_image],
            outputs=[target_dir_output, log_output],
        )

        # Dynamically update intrinsic parameter visibility when camera selection changes.
        camera_model.change(
            fn=update_parameters,
            inputs=camera_model,
            outputs=[fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, t1, t2, hfov],
        )

        demo.launch(show_error=True)
