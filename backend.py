from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import tempfile
from temple_structure_analyzer import TempleStructureAnalyzer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/detect-image', methods=['POST'])
def detect_image():
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("detect_image")
    try:
        if 'image' not in request.files:
            logger.error("No image uploaded in request.files")
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        logger.info(f"Saving uploaded image to {img_path}")
        file.save(img_path)

        if not os.path.exists(img_path):
            logger.error(f"File not saved at {img_path}")
            return jsonify({'error': 'File not saved'}), 500

        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"cv2.imread failed to read image at {img_path}")
            return jsonify({'error': 'Failed to read image'}), 500

        logger.info("Running image detection pipeline")
        from integratedimgdetection import extract_significant_objects, branch_and_bound_feature_matching, backtracking_segmentation
        object_img = extract_significant_objects(img)
        logger.info("Extracted significant objects")
        matched_img = branch_and_bound_feature_matching(object_img, img)
        if matched_img is None:
            logger.error("Feature matching failed")
            return jsonify({'error': 'Feature matching failed'}), 500
        logger.info("Feature matching succeeded")
        final_output = backtracking_segmentation(matched_img)
        logger.info("Backtracking segmentation completed")

        result_path = os.path.join(RESULT_FOLDER, 'result_image.png')
        success = cv2.imwrite(result_path, final_output)
        if not success or not os.path.exists(result_path):
            logger.error(f"cv2.imwrite failed or file not found at {result_path}")
            return jsonify({'error': 'Failed to save result image'}), 500
        logger.info(f"Result image saved at {result_path}")
        return send_file(result_path, mimetype='image/png')
    except Exception as e:
        logger.exception(f"Exception in detect_image: {str(e)}")
        return jsonify({'error': f'Exception: {str(e)}'}), 500

@app.route('/detect-video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    output_path = os.path.join(RESULT_FOLDER, 'result_video.mp4')
    analyzer = TempleStructureAnalyzer(yolo_model_path="yolov8n.pt", use_windows=False)
    results = analyzer.process_video(video_path, output_path)

    # Return video and summary
    return jsonify({
        'video_url': '/get-result-video',
        'summary': results
    })

@app.route('/get-result-video')
def get_result_video():
    video_path = os.path.join(RESULT_FOLDER, 'result_video.mp4')
    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)