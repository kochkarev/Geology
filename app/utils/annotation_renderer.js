const fs = window.require('fs');
const UPNG = require('upng-js');


class ActiveImageWithAnnotationRenderer {

    constructor(img, canvClassAnnoTmp, canvInstAnnoTmp, canvClassAnno, canvInstAnno, labelsMap) {
        this.imgElem = img;
        this.canvClassAnnoTmp = canvClassAnnoTmp;
        this.canvInstAnnoTmp = canvInstAnnoTmp;
        this.canvClassAnno = canvClassAnno;
        this.canvInstAnno = canvInstAnno;

        this.labelsMap = labelsMap;

        this.ctxClassAnnoTmp = this.canvClassAnnoTmp.getContext('2d');
        this.ctxInstAnnoTmp = this.canvInstAnnoTmp.getContext('2d');
        this.ctxClassAnno = this.canvClassAnno.getContext('2d');
        this.ctxInstAnno = this.canvInstAnno.getContext('2d');
        
        this.imgStruct = null;
        this.classAnno = null;
        this.instAnno = null;
        this.prevInstId = null;
        this.scale = 1;

        this.fw = null;
        this.fh = null;
        this.w = null;
        this.h = null;

        this.instColorized = new Map();
    }

    changeContext(imgStruct) {
        this.imgStruct = imgStruct;
        let _img = fs.readFileSync(imgStruct.filePath).toString('base64');
        this.imgElem.src = `data:image/jpg;base64,${_img}`;
        this.instAnno = null;
        this.prevInstId = null;
        this.clearInstAnno();
        this.instColorized.clear();
        ipcRenderer.send('active-image-update', imgStruct.id);    
        this.imgElem.onload = () => {
            this.recalcSize();
            this.renderImage();
            this.loadClassAnno(imgStruct);
            this.renderClassAnno();
        }
    }

    loadClassAnno(imgStruct, alpha=150) {
        // transform mask to colorized annotation
        let mask = UPNG.decode(fs.readFileSync(imgStruct.maskPath));
        console.log(this.fw, this.fh, mask.width, mask.height);
        this.classAnno = colorizeClassAnno(mask, this.labelsMap, alpha=alpha);
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
            if (!this.instColorized.has(inst.id)) {
                this.instColorized.set(inst.id, colorizeInstMask(inst, this.labelsMap, alpha));
            }
            let colorized = this.instColorized.get(inst.id);
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

    instAnnoUpdateFromMain(anno) {
        if (anno.imgId !== this.imgStruct.id)
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


function colorizeClassAnno(png, labelsMap, alpha) {
    let anno = new Uint8ClampedArray(png.width * png.height * 4);
    for (i = 0; i < png.width * png.height; i++) {
        let v = png.data[3 * i];
        let color = labelsMap[v].color;
        anno[4 * i] = color[0];
        anno[4 * i + 1] = color[1];
        anno[4 * i + 2] = color[2];
        if (v > 0)
            anno[4 * i + 3] = alpha;
    }
    return anno;    
}

function colorizeInstMask(inst, labelsMap, alpha) {
    const color = labelsMap[inst.class].color;
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


module.exports = {ActiveImageWithAnnotationRenderer: ActiveImageWithAnnotationRenderer};