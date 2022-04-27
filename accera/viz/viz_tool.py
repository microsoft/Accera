#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Authors: Alex Soustek
####################################################################################################

# I will host a local web server to host the tool written in JS (three.js) and provide
# some APIs which create/manage folders for the visualization tool, i may also be used to store images and combine them into a
# video vis FFMPEG bindings wooooo!

import webbrowser
import os
import logging
import json
import bottle
import glob
import re
import shutil
import argparse
import fnmatch
import urllib
import threading
import time
from binascii import a2b_base64
from bottle import static_file, run, route, get, post, redirect, response, request
from os.path import relpath

try:
    import ffmpeg
except:
    pass

parser = argparse.ArgumentParser(description='A tool to help create visualizations for iteration spaces.')
parser.add_argument('--port', nargs='?', const=8000, default=8000, type=int, help='Specify the port for the webserver')

parser.add_argument(
    '--automate', action='store_true', help='Should the server kick off an automated run of visualizations on startup?'
)

parser.add_argument('--video', action='store_true', help='Should the automated run store video?')

parser.add_argument('--files', type=str, help='Specify the filename or unix glob-style pattern to match.')

args = parser.parse_args()
print(args)

PORT = args.port
AUTOMATE = args.automate
RENDER_VIDEO = args.video
FILE_PATTERN = args.files

# Typically 102400, but pictures can be big yo
bottle.BaseRequest.MEMFILE_MAX = 1024 * 1000


# Path and file helpers
def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_path_recursively(path):
    if os.path.exists(path):
        print('Deleting ' + str(path))
        shutil.rmtree(path)
    else:
        print('No folder to delete at ' + str(path))


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def replace_file(filename, binary_contents):
    delete_file(filename)
    try:
        fd = open(filename, 'wb')
        fd.write(binary_contents)
        fd.close()
        return True
    except OSError:
        return False


def make_name_path_safe(name):
    name = re.sub(r'[^a-zA-Z0-9 -_]', '', name)
    name = name.replace(' ', '_')
    return name


def get_highest_filename(path):
    dir = os.listdir(path)

    if len(dir) == 0:
        return 0

    files = os.listdir(path)
    files_png = [i for i in files if i.endswith('.png')]

    def extract_number(f):
        s = re.findall("\d+$", f)
        return (int(s[0]) if s else -1, f)

    highest_filename = max(files_png, key=extract_number).replace('.png', '')
    return int(highest_filename)


def create_sequence_filename(filenum):
    return f"{filenum:0>6}"


# App dirs
static_dir = os.path.join(os.path.dirname(__file__), 'static')
viz_input_dir = os.path.join(os.path.dirname(__file__), 'visualizations', 'input')
viz_output_dir = os.path.join(os.path.dirname(__file__), 'visualizations', 'output')


def get_sequence_path(viz_name, sequence_name):
    return os.path.join(viz_output_dir, viz_name, 'sequences', sequence_name)


# Create output path
ensure_path_exists(viz_output_dir)


# Server paths
@route('/')
def serve_default():
    return redirect("/static/index.html")


@route('/static/<filename:path>')
def serve_static(filename):
    print(filename)
    response = static_file(filename, root=static_dir)
    if (filename.find('.js') != -1):
        response.headers['Content-Type'] = 'application/javascript'
    print(response.headers['Content-Type'])
    return response


@route('/visualizations/<filename:path>')
def serve_visualizations(filename):
    response = static_file(filename, root=viz_input_dir)
    response.set_header("Cache-Control", "no-store")
    response.headers['Content-Type'] = 'application/javascript'
    return response


@get('/api/list_visualizations')
def list_visualizations():
    response.headers['Content-Type'] = 'application/json'
    viz_files = [relpath(f, viz_input_dir) for f in glob.iglob(f"{viz_input_dir}/**/*.js", recursive=True)]
    viz_files.remove('common.js')
    viz_files.sort(reverse=True)    # TODO: remove reverse sort
    return json.dumps({'visualizations': viz_files})


@post('/api/save_image')
def save_image():
    response.headers['Content-Type'] = 'application/json'
    frame_name = make_name_path_safe(request.json['frame_name'])
    viz_name = make_name_path_safe(request.json['viz_name'])
    output_directory = os.path.join(viz_output_dir, viz_name)
    ensure_path_exists(output_directory)

    file_path = os.path.join(output_directory, frame_name + ".png")
    print('Saving frame ' + frame_name + ' for viz ' + viz_name + ' to: ' + str(file_path))

    trimmed_data = request.json['frame'].replace('data:image/png;base64,', '')
    binary_data = a2b_base64(trimmed_data)
    return json.dumps({'success': replace_file(file_path, binary_data)})


@post('/api/start_sequence')
def start_sequence():
    response.headers['Content-Type'] = 'application/json'
    viz_name = make_name_path_safe(request.json['viz_name'])
    sequence_name = make_name_path_safe(request.json['sequence_name'])
    fps = int(request.json['sequence_fps'])

    output_directory = get_sequence_path(viz_name, sequence_name)
    delete_path_recursively(output_directory)
    ensure_path_exists(output_directory)

    print('Starting sequence named ' + sequence_name + ' for viz ' + viz_name + ' saving to: ' + str(output_directory))

    return json.dumps({'success': True})


@post('/api/add_to_sequence')
def add_to_sequence():
    response.headers['Content-Type'] = 'application/json'
    viz_name = make_name_path_safe(request.json['viz_name'])
    sequence_name = make_name_path_safe(request.json['sequence_name'])
    fps = int(request.json['sequence_fps'])

    output_directory = get_sequence_path(viz_name, sequence_name)
    ensure_path_exists(output_directory)

    trimmed_data = request.json['frame'].replace('data:image/png;base64,', '')
    binary_data = a2b_base64(trimmed_data)

    current_highest_frame = get_highest_filename(output_directory)
    duration_one_frame = 1000 // fps
    num_frames = 1

    if 'frame_duration' in request.json:
        frame_duration = int(request.json['frame_duration'])
        num_frames = frame_duration // duration_one_frame

    print('Existing highest frame ' + str(current_highest_frame) + ' adding ' + str(num_frames) + ' frames')
    success = True
    for f in range(1, num_frames + 1):
        file_path = os.path.join(output_directory, create_sequence_filename(current_highest_frame + f) + ".png")
        if replace_file(file_path, binary_data) == False:
            success = False
            break

    return json.dumps({'success': success})


@post('/api/finish_sequence')
def finish_sequence():
    response.headers['Content-Type'] = 'application/json'
    viz_name = make_name_path_safe(request.json['viz_name'])
    sequence_name = make_name_path_safe(request.json['sequence_name'])
    fps = int(request.json['sequence_fps'])

    input_directory = get_sequence_path(viz_name, sequence_name)
    output_file = os.path.join(viz_output_dir, viz_name, sequence_name + '.mp4')
    delete_file(output_file)

    input_glob = os.path.join(input_directory, '%06d.png')
    print(str(input_glob))
    print(str(output_file))
    print('Rendering sequence ' + str(sequence_name) + ' to video at ' + str(fps) + ' fps')

    try:
        (ffmpeg.input(str(input_glob), framerate=fps).output(str(output_file), pix_fmt='yuv420p', vb='20M').run())
    except Exception as e:
        print('Failed to invoke ffmpeg. Try "pip install ffmpeg-python" to install this dependency')
        print(str(e))
        return json.dumps({'success': False})

    return json.dumps({'success': True})


def get_automated_file_list():
    viz_files = [relpath(f, viz_input_dir) for f in glob.iglob(f"{viz_input_dir}/**/*.js", recursive=True)]
    viz_files.remove('common.js')
    filtered_viz_files = [file for file in viz_files if fnmatch.fnmatch(file, FILE_PATTERN)]

    return filtered_viz_files


#http://localhost:8000/static/index.html?selected_viz=contained_split_keyframes.js&save_through_browser=false&enable_video=true&enable_image=true&automated_run=true&remaining_files=?
def attempt_automated_run():
    file_list = get_automated_file_list()
    if len(file_list) == 0:
        print('No files match the pattern ' + str(FILE_PATTERN) + ' aborting automated run')
        return

    print('Automated run will include: ' + str(len(file_list)) + ' files')

    first_file = file_list.pop(0)
    file_list_csv = ','.join(file_list)
    params = {
        'selected_viz': first_file,
        'save_through_browser': 'false',
        'enable_image': 'true',
        'automated_run': 'true',
        'enable_video': str(RENDER_VIDEO).lower(),
        'remaining_files': file_list_csv
    }
    url_params = urllib.parse.urlencode(params)
    automated_url = urllib.parse.urlunparse(
        ('http', 'localhost:' + str(PORT), '/static/index.html', None, url_params, None)
    )
    print(automated_url)

    def open_browser(url):
        time.sleep(2)
        logging.info("Attempting to open browser for automated run")
        webbrowser.open(url)

    browser_thread = threading.Thread(target=open_browser, args=[automated_url])
    browser_thread.start()


# Attempt automated run if requested
if AUTOMATE:
    attempt_automated_run()

# Start server
print("Serving at port", PORT)
run(host='localhost', port=PORT, debug=True)
