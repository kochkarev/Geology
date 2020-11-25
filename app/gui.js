const {ipcRenderer} = require('electron');
const fs = window.require('fs');
const UPNG = require('upng-js');

const {ShallowImageList} = require('./utils/structs_ui')

const canvClassAnno = document.getElementById('canv-class-anno');
const main_image_zone = document.getElementById('main-image-zone');

let mouseDown = false;

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


class ActiveImageWithAnnotationRenderer {

    constructor(img, canvClassAnnoTmp, canvInstAnnoTmp, canvClassAnno, canvInstAnno) {
        this.imgElem = img;
        this.canvClassAnnoTmp = canvClassAnnoTmp;
        this.canvInstAnnoTmp = canvInstAnnoTmp;
        this.canvClassAnno = canvClassAnno;
        this.canvInstAnno = canvInstAnno;

        this.ctxClassAnnoTmp = this.canvClassAnnoTmp.getContext('2d');
        this.ctxInstAnnoTmp = this.canvInstAnnoTmp.getContext('2d');
        this.ctxClassAnno = this.canvClassAnno.getContext('2d');
        this.ctxInstAnno = this.canvInstAnno.getContext('2d');
        
        this.activeIdx = null;
        this.imgStruct = null;
        this.classAnno = null;
        this.instAnno = null;
        this.prevInstId = null;
        this.scale = 1;

        this.fw = null;
        this.fh = null;
        this.w = null;
        this.h = null;
    }

    changeContext(imgStruct, idx) {
        this.activeIdx = idx;
        this.imgStruct = imgStruct;
        let _img = fs.readFileSync(imgStruct.filePath).toString('base64');
        this.imgElem.src = `data:image/jpg;base64,${_img}`;
        this.instAnno = null;
        this.prevInstId = null;
        this.clearInstAnno();
        ipcRenderer.send('active-image-update', idx);    
        this.imgElem.onload = () => {
            this.recalcSize();
            this.renderImage();
            this._loadClassAnno(imgStruct);
            this.renderClassAnno();
        }
    }

    _loadClassAnno(imgStruct, alpha=150) {
        // transform mask to colorized annotation
        let mask = UPNG.decode(fs.readFileSync(imgStruct.maskPath));
        console.log(this.fw, this.fh, mask.width, mask.height);
        this.classAnno = colorizeClassAnno(mask, annoColorsArr, alpha=alpha);
        // render to temporary canvas
        this.canvClassAnnoTmp.width = this.fw;
        this.canvClassAnnoTmp.height = this.fh;
        this.ctxClassAnnoTmp.putImageData(new ImageData(this.classAnno, this.fw, this.fh), 0, 0);
    }

    recalcSize() {
        this.fw = this.imgElem.naturalWidth;
        this.fh = this.imgElem.naturalHeight;
        this.w = Math.round(this.scale * this.imgElem.naturalWidth);
        this.h = Math.round(this.scale * this.imgElem.naturalHeight);
    }

    renderImage() {
        this.imgElem.width = this.w;
        this.imgElem.height = this.h;
    }

    renderClassAnno() {
        this.canvClassAnno.width = this.w;
        this.canvClassAnno.height = this.h;
        this.ctxClassAnno.clearRect(0, 0, this.w, this.h);
        this.ctxClassAnno.drawImage(this.canvClassAnnoTmp, 0, 0, this.w, this.h);
    }

    clearInstAnno() {
        this.canvInstAnno.width = this.w;
        this.canvInstAnno.height = this.h;
        this.ctxInstAnno.clearRect(0, 0, this.w, this.h);
    }

    prepareInstAnno(inst, alpha=255) {
        this.ctxInstAnnoTmp.clearRect(0, 0, this.fw, this.fh);
        if (inst) {
            let colorized = colorizeInstMask(inst, annoColorsArr, alpha);
            this.ctxInstAnnoTmp.putImageData(new ImageData(colorized, inst.w, inst.h), inst.x, inst.y);
        }
    }

    renderInstAnno() {
        this.clearInstAnno()
        this.ctxInstAnno.drawImage(this.canvInstAnnoTmp, 0, 0, this.w, this.h);                   
    }

    changeScale(code, step=0.1) {
        if (code === 'BracketRight') {
            this.scale = Math.min(this.scale + step, 2);
        }
        else if (code === 'BracketLeft') {
            this.scale = Math.max(this.scale - step, 0.1);
        }
        this.recalcSize();
        this.renderImage();
        this.renderClassAnno();
        this.renderInstAnno();
    }

    instAnnoUpdateFromMain(idx, anno) {
        if (idx !== this.activeIdx)
            return;
        this.instAnno = anno;
        this.canvInstAnnoTmp.width = this.fw;
        this.canvInstAnnoTmp.height = this.fh;    
    }

    cursorMove(x, y) {
        let xs = Math.round(x / this.scale);
        let ys = Math.round(y / this.scale);

        if (this.instAnno?.instMap) {
            let instId = this.instAnno.instMap.data[(ys * this.instAnno.instMap.w + xs) * 3];
            if (instId !== this.prevInstId) {
                this.prepareInstAnno(this.instAnno.instances[instId - 1]);
                this.renderInstAnno();
                this.prevInstId;
            }
        }
    }
}


let R = new ActiveImageWithAnnotationRenderer(
    document.getElementById('img'),
    document.getElementById('canv-anno-tmp'),
    document.getElementById('canv-inst-tmp'),
    document.getElementById('canv-class-anno'),
    document.getElementById('canv-inst-anno'),
);



main_image_zone.addEventListener('mousemove', (e) => {
    let rect = canvClassAnno.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    if (mouseDown) {
    }
    else {
        R.cursorMove(x, y);
    }
});

let imgStructs = new ShallowImageList(
    document.getElementById('files-list'),
    (struct, index) => {
        R.changeContext(struct, index);
    }
);

document.addEventListener('keydown', (e) => {
    if (e.code === 'ArrowDown' || e.code === 'ArrowUp')
        imgStructs.moveSelection(e.code);
    else if (e.code === 'BracketLeft' || e.code === 'BracketRight')
        R.changeScale(e.code);    
});

imgStructs.listGroupHTML.addEventListener('click', e => imgStructs.selectClick(e));

document.getElementById('btn_st1').addEventListener('click', () => runClick(step=1));
document.getElementById('btn_st2').addEventListener('click', () => runClick(step=2));
document.getElementById('btn_stop').addEventListener('click', () => ipcRenderer.send('stop-algo'));

ipcRenderer.on('files-added', (e, data) => {
    imgStructs.addFiles(data);
});

ipcRenderer.on('anno-loaded', (event, idx, anno) => {
    console.log(`Hey! Got instance annotation update ${idx}`);
    R.instAnnoUpdateFromMain(idx, anno);
});

function colorizeClassAnno(png, colors, alpha) {
    let anno = new Uint8ClampedArray(png.width * png.height * 4);
    for (i = 0; i < png.width * png.height; i++) {
        let v = png.data[3 * i];
        anno[4 * i] = colors[v][0];
        anno[4 * i + 1] = colors[v][1];
        anno[4 * i + 2] = colors[v][2];
        if (v > 0)
            anno[4 * i + 3] = alpha;
    }
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
