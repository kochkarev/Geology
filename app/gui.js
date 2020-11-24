const {ipcRenderer} = require('electron');
const fs = window.require('fs');
const path = window.require('path');

const {ShallowImgStruct, ShallowImageList} = require('./utils/structs_ui')

const main_image_zone = document.getElementById('main-image-zone');
const img = document.getElementById('img');
const canv_class_anno = document.getElementById('canv-class-anno');
// const canv_brush = document.getElementById('canv-brush');
// const ctx_anno = canv_anno.getContext('2d');
// const ctx_brush = canv_brush.getContext('2d');

let selectedClass = 'object';
let [prev_x, prev_y] = [0, 0]

let mouseRad = 5;
let mouseDown = false;

const annoColors = {'object': 'rgb(255, 255, 0)', background: 'rgb(0, 255, 0)'};
const annoColorsArr = [[0, 0, 0], [255, 255, 0], [0, 255, 0]];

const supportLevels = ['off', 'CNN', 'CNN+dist']
let selectedSupportLevel = 1

document.getElementById('r1').addEventListener('click', () => {
    selectedClass = 'object';
});
document.getElementById('r2').addEventListener('click', () => {
    selectedClass = 'background';
});

document.getElementById('selector-algo').onchange = () => {
    e = document.getElementById('selector-algo');
    selectedSupportLevel = e.options[e.selectedIndex].value
};

updateCanvSize();



function updateBrushSize(code, step = 1) {
    console.log('update brush');
    if (code === 'BracketRight') {
        mouseRad += step;
    } else if (code === 'BracketLeft') {
        mouseRad -= step;
    }
    console.log(mouseRad);
    renderCursorMove(prev_x, prev_y, prev_x, prev_y);
}

function updateCanvSize() {
    canv_class_anno.width = img.width;
    canv_class_anno.height = img.height;
    // canv_brush.width = img.width;
    // canv_brush.height = img.height;
}

function updateImg(imgStruct, idx) {
    let _img = fs.readFileSync(imgStruct.filePath).toString('base64');
    img.src = `data:image/jpg;base64,${_img}`;
    img.onload = function(){
        updateCanvSize();
        // if (imgStructs.inputAnnotations[index] !== null)
            // ctx_anno.putImageData(imgStructs.inputAnnotations[imgStructs.activeIdx], 0, 0);
    }
    ipcRenderer.send('active-image-update', idx);
}


function renderCursorMove(x1, y1, x2, y2, alpha=1.0) {
    // ctx_brush.clearRect(x1 - mouseRad * 2, y1 - mouseRad * 2, 4 * mouseRad, 4 * mouseRad);
    // ctx_brush.globalAlpha = alpha;
    // ctx_brush.fillStyle = annoColors[selectedClass];
    // ctx_brush.fillRect(x2 - mouseRad, y2 - mouseRad, 2 * mouseRad, 2 * mouseRad);
}

function getCanvasCoords(e) {
    // var rect = canv_brush.getBoundingClientRect();
    // let x = e.clientX - rect.left;
    // let y = e.clientY - rect.top;
    // return [x, y];
    return [0, 0];
}

main_image_zone.addEventListener('mousedown', (e) => {
    mouseDown = true;
    let [x, y] = getCanvasCoords(e);
    renderScribble(x, y, x, y);
})

main_image_zone.addEventListener('mouseup', (e) => {
    mouseDown = false;
    imgStructs.inputAnnotations[imgStructs.activeIdx] = ctx_anno.getImageData(0, 0, canv_class_anno.width, canv_class_anno.height);
})

main_image_zone.addEventListener('mousemove', (e) => {
    let [x, y] = getCanvasCoords(e);
    if (mouseDown)
        renderScribble(prev_x, prev_y, x, y);
    else
        renderCursorMove(prev_x, prev_y, x, y);
    [prev_x, prev_y] = [x, y];

    ipcRenderer.send('mouse-move', [x, y]);
});

let imgStructs = new ShallowImageList(
    document.getElementById('files-list'),
    (s, i) => updateImg(s, i)
);

document.addEventListener('keydown', (e) => {
    if (e.code === 'ArrowDown' || e.code === 'ArrowUp')
        imgStructs.moveSelection(e.code);
    else if (e.code === 'BracketLeft' || e.code === 'BracketRight')
        updateBrushSize(e.code);
});

imgStructs.listGroupHTML.addEventListener('click', e => imgStructs.selectClick(e));

document.getElementById('btn_st1').addEventListener('click', () => runClick(step=1));
document.getElementById('btn_st2').addEventListener('click', () => runClick(step=2));
document.getElementById('btn_stop').addEventListener('click', () => ipcRenderer.send('stop-algo'));

ipcRenderer.on('files-added', (e, data) => {
    imgStructs.addFiles(data);
});

ipcRenderer.on('anno-update', (event, anno) => {
    let img_data = new ImageData(decompressToColoredAnno(anno, annoColorsArr, alpha=255), canv_class_anno.width, canv_class_anno.height);
    ctx_anno.putImageData(img_data, 0, 0);
});

function decompressToColoredAnno(anno, colors, alpha) {
    let anno_decompressed = new Uint8ClampedArray(anno.length * 4);
    for (i = 0; i < anno.length; i ++) {
        let v = anno[i];
        anno_decompressed[4 * i] = colors[v][0];
        anno_decompressed[4 * i + 1] = colors[v][1];
        anno_decompressed[4 * i + 2] = colors[v][2];
        if (v > 0)
            anno_decompressed[4 * i + 3] = alpha;
    }
    return anno_decompressed;    
}

function compressColoredAnno(anno, colors) {
    let anno_compressed = new Uint8Array(anno.length / 4);
    let dist = new Uint16Array(colors.length);
    for (i = 0; i < anno.length; i += 4) {
        let r = anno[i];
        let g = anno[i + 1];
        let b = anno[i + 2];        
        for (j = 0; j < colors.length; j++) {
            dist[j] = (r - colors[j][0]) ** 2 + (g - colors[j][1]) ** 2 + (b - colors[j][2]) ** 2;
        }
        anno_compressed[i / 4] = dist.indexOf(Math.min(...dist));
    }
    return anno_compressed;
}


function runClick(step) {
    let anno = ctx_anno.getImageData(0, 0, canv_class_anno.width, canv_class_anno.height).data;
    let anno_compact = compressColoredAnno(anno, annoColorsArr);
    let header = {
        'type': 'anno',
        'width': canv_class_anno.width,
        'height': canv_class_anno.height,
        'image_path': imgStructs.filePaths[imgStructs.activeIdx],
        'support': supportLevels[selectedSupportLevel - 1],
        'step': step
    };
    ipcRenderer.send('anno', [header, anno_compact]);
}
