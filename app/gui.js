const {ipcRenderer} = require('electron');
const fs = window.require('fs');
const UPNG = require('upng-js');

const {ShallowImgStruct, ShallowImageList} = require('./utils/structs_ui')

const main_image_zone = document.getElementById('main-image-zone');
const img = document.getElementById('img');
const canv_class_anno = document.getElementById('canv-class-anno');
const canv_inst_anno = document.getElementById('canv-inst-anno');
const ctx_class_anno = canv_class_anno.getContext('2d');
const ctx_inst_anno = canv_inst_anno.getContext('2d');
// const ctx_brush = canv_brush.getContext('2d');

// let selectedClass = 'object';
// let [prev_x, prev_y] = [0, 0]

// let mouseRad = 5;

let inst_anno = null;
let inst_id = -1;
let prev_inst_id = -1;

let scale = 0.5;

let mouseDown = false;

let activeAnnotation = null;

const annoColors = {
    1: 'rgb(255, 255, 0)',
    2: 'rgb(0, 255, 255)',
    3: 'rgb(0, 255, 0)',
    4: 'rgb(0, 0, 255)',
    5: 'rgb(255, 0, 0)',
    6: 'rgb(100, 255, 0)',
    7: 'rgb(255, 100, 100)',
};
const annoColorsArr = [
    [0, 0, 0],
    [255, 255, 0],
    [0, 255, 255],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 0],
    [100, 255, 0],
    [255, 100, 100],
    [100, 255, 255],
    [200, 50, 80],
    [80, 50, 200],
    [20, 120, 255],
    [120, 20, 255],

];


function updateCanvSize(img) {
    canv_class_anno.width = img.width;
    canv_class_anno.height = img.height;
    canv_inst_anno.width = img.width;
    canv_inst_anno.height = img.height;
}

function updateImg(imgStruct, idx) {
    let _img = fs.readFileSync(imgStruct.filePath).toString('base64');
    img.src = `data:image/jpg;base64,${_img}`;
    img.onload = function(){
        updateCanvSize(img);
        let mask = UPNG.decode(fs.readFileSync(imgStruct.maskPath));
        let c = colorizeClassAnno(mask, annoColorsArr, alpha=150);
        ctx_class_anno.putImageData(new ImageData(c, mask.width, mask.height), 0, 0);
    }
    ipcRenderer.send('active-image-update', idx);
}

function updateInstAnnotation(anno) {
    inst_anno = anno;
}

function updateInstMask(inst) {
    console.log(inst);
    ctx_inst_anno.clearRect(0, 0, canv_inst_anno.width, canv_inst_anno.height);
    if (inst) {
        let colorized = colorizeInstMask(inst, annoColorsArr, 255);
        ctx_inst_anno.putImageData(new ImageData(colorized, inst.w, inst.h), inst.x, inst.y);
    }
}

function renderCursorMove(x, y, alpha=1.0) {
    if (inst_anno) {
        if (!inst_anno.instMap) {
            return;
        }
        let inst_id = inst_anno.instMap.data[(y * inst_anno.instMap.w + x) * 3];
        if (inst_id !== prev_inst_id) {
            updateInstMask(inst_anno.instances[inst_id - 1]);
            // if (inst_id > 0) {
            //     let inst = inst_anno.instances[inst_id - 1];
            // }
            prev_inst_id = inst_id;
        }
        // let inst = inst_anno.instances[inst_id];
        // console.log(inst);
        // let inst = inst_anno.getInstByCoords(x, y);
        // console.log(inst);
    }
    // ctx_brush.clearRect(x1 - mouseRad * 2, y1 - mouseRad * 2, 4 * mouseRad, 4 * mouseRad);
    // ctx_brush.globalAlpha = alpha;
    // ctx_brush.fillStyle = annoColors[selectedClass];
    // ctx_brush.fillRect(x2 - mouseRad, y2 - mouseRad, 2 * mouseRad, 2 * mouseRad);
}

function getCanvasCoords(e) {
    var rect = canv_class_anno.getBoundingClientRect();
    let x = Math.round(e.clientX - rect.left);
    let y = Math.round(e.clientY - rect.top);
    return [x, y];
}

// main_image_zone.addEventListener('mousedown', (e) => {
//     mouseDown = true;
//     let [x, y] = getCanvasCoords(e);
//     renderScribble(x, y, x, y);
// })

// main_image_zone.addEventListener('mouseup', (e) => {
//     mouseDown = false;
//     imgStructs.inputAnnotations[imgStructs.activeIdx] = ctx_anno.getImageData(0, 0, canv_class_anno.width, canv_class_anno.height);
// })

main_image_zone.addEventListener('mousemove', (e) => {
    let [x, y] = getCanvasCoords(e);
    if (mouseDown) {
        // renderScribble(prev_x, prev_y, x, y);
    }
    else {
        renderCursorMove(x, y);
    }
    [prev_x, prev_y] = [x, y];
});

let imgStructs = new ShallowImageList(
    document.getElementById('files-list'),
    (s, i) => updateImg(s, i)
);

document.addEventListener('keydown', (e) => {
    if (e.code === 'ArrowDown' || e.code === 'ArrowUp')
        imgStructs.moveSelection(e.code);
    // else if (e.code === 'BracketLeft' || e.code === 'BracketRight')
    //     updateBrushSize(e.code);
});

imgStructs.listGroupHTML.addEventListener('click', e => imgStructs.selectClick(e));

document.getElementById('btn_st1').addEventListener('click', () => runClick(step=1));
document.getElementById('btn_st2').addEventListener('click', () => runClick(step=2));
document.getElementById('btn_stop').addEventListener('click', () => ipcRenderer.send('stop-algo'));

ipcRenderer.on('files-added', (e, data) => {
    imgStructs.addFiles(data);
});

ipcRenderer.on('anno-loaded', (event, idx, anno) => {
    console.log(`Hey! Got annotation update ${idx}`);
    if (idx == imgStructs.activeIdx) {
        updateInstAnnotation(anno);
    }
});

function colorizeClassAnno(png, colors, alpha) {
    // console.log(png);
    let anno = new Uint8ClampedArray(png.width * png.height * 4);
    for (i = 0; i < png.width * png.height; i++) {
        let v = png.data[3 * i];
        // console.log(v);
        anno[4 * i] = colors[v][0];
        anno[4 * i + 1] = colors[v][1];
        anno[4 * i + 2] = colors[v][2];
        if (v > 0)
            anno[4 * i + 3] = alpha;
    }
    // console.log(anno);
    return anno;    
}

function colorizeInstMask(inst, colors, alpha) {
    let color = colors[inst.class];
    let anno = new Uint8ClampedArray(inst.w * inst.h * 4);
    for (i = 0; i < inst.w * inst.h; i ++) {
        if (inst.mask[i] > 0) {
            anno[4 * i] = color[0];
            anno[4 * i + 1] = color[1];
            anno[4 * i + 2] = color[2];
            anno[4 * i + 3] = alpha;
        }
    }
    return anno;    
}
