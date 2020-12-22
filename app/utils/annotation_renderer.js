const fs = window.require('fs');


class ActiveImageWithAnnotationRenderer {

    constructor(img, canvAnnoSemTmp, canvAnnoInstTmp, canvAnnoSem, canvAnnoInst, statisticsTextElem, labelsMap, scaleCoeff) {
        this.imgElem = img;
        this.canvAnnoSemTmp = canvAnnoSemTmp;
        this.canvAnnoInstTmp = canvAnnoInstTmp;
        this.canvAnnoSem = canvAnnoSem;
        this.canvAnnoInst = canvAnnoInst;
        this.statisticsTextElem = statisticsTextElem;

        this.labelsMap = labelsMap;
        this.scaleCoeff = scaleCoeff;

        this.ctxAnnoSemTmp = this.canvAnnoSemTmp.getContext('2d');
        this.ctxAnnoInstTmp = this.canvAnnoInstTmp.getContext('2d');
        this.ctxAnnoSem = this.canvAnnoSem.getContext('2d');
        this.ctxAnnoInst = this.canvAnnoInst.getContext('2d');
        
        this.imgStruct = null;
        this.annoSemColorized = null;
        this.annoInst = null;
        this.prevInstId = null;
        this.scale = 1;

        this.w = null;
        this.h = null;

        this.instColorized = new Map();
    }

    changeContext(imgStruct) {
        this.imgStruct = imgStruct;
        let _img = fs.readFileSync(imgStruct.imagePath).toString('base64');
        this.imgElem.src = `data:image/jpg;base64,${_img}`;
        this.annoSem = null;
        this.annoSemColorized = null;
        this.annoInst = null;
        this.prevInstId = null;
        this.clearAnnoInst();
        this.instColorized.clear();
        ipcRenderer.send('active-image-update', imgStruct.id);    
        this.imgElem.onload = () => {
            this.recalcSize();
            this.renderImage();
            this.loadAnnoSem();
            this.renderAnnoSem();
        }
    }

    loadAnnoSem(alpha=150) {
        // transform mask to colorized annotation
        this.annoSem = this.imgStruct.annoSemanticGT;
        this.annoSemColorized = getColorizeAnnoSem(this.imgStruct.annoSemanticGT, this.imgStruct.w, this.imgStruct.h, this.labelsMap, alpha=alpha);
        // render to temporary canvas
        this.canvAnnoSemTmp.width = this.imgStruct.w;
        this.canvAnnoSemTmp.height = this.imgStruct.h;
        this.ctxAnnoSemTmp.putImageData(new ImageData(this.annoSemColorized, this.imgStruct.w, this.imgStruct.h), 0, 0);
    }

    recalcSize() {
        this.w = Math.round(this.scale * this.imgElem.naturalWidth);
        this.h = Math.round(this.scale * this.imgElem.naturalHeight);
    }

    renderImage() {
        this.imgElem.width = this.w;
        this.imgElem.height = this.h;
    }

    renderAnnoSem() {
        this.canvAnnoSem.width = this.w;
        this.canvAnnoSem.height = this.h;
        this.ctxAnnoSem.clearRect(0, 0, this.w, this.h);
        this.ctxAnnoSem.drawImage(this.canvAnnoSemTmp, 0, 0, this.w, this.h);
    }

    clearAnnoInst() {
        this.canvAnnoInst.width = this.w;
        this.canvAnnoInst.height = this.h;
        this.ctxAnnoInst.clearRect(0, 0, this.w, this.h);
    }

    prepareAnnoInst(inst, alpha=255) {
        this.ctxAnnoInstTmp.clearRect(0, 0, this.imgStruct.w, this.imgStruct.h);
        if (inst) {
            if (!this.instColorized.has(inst.id)) {
                this.instColorized.set(inst.id, colorizeAnnoInst(inst, this.labelsMap, alpha));
            }
            let colorized = this.instColorized.get(inst.id);
            this.ctxAnnoInstTmp.putImageData(new ImageData(colorized, inst.w, inst.h), inst.x, inst.y);
        }
    }

    renderAnnoInst() {
        this.clearAnnoInst()
        this.ctxAnnoInst.drawImage(this.canvAnnoInstTmp, 0, 0, this.w, this.h);                   
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
        this.renderAnnoSem();
        this.renderAnnoInst();
    }

    annoInstUpdateFromMain(anno) {
        if (anno.imgId !== this.imgStruct.id)
            return;
        this.annoInst = anno;
        this.canvAnnoInstTmp.width = this.imgStruct.w;
        this.canvAnnoInstTmp.height = this.imgStruct.h;    
    }

    cursorMove(x, y) {
        let xs = Math.round(x / this.scale);
        let ys = Math.round(y / this.scale);

        if (!this.annoInst && this.annoSem) {
            let classLabel = getClassLabel(this.annoSem, this.imgStruct.w, this.imgStruct.h, this.labelsMap, xs, ys);
            this.statisticsTextElem.innerHTML = `class: ${classLabel}`;
        }

        if (this.annoInst?.instMap) {
            let instId = this.annoInst.instMap.data[(ys * this.annoInst.instMap.w + xs) * 3];
            if (instId !== this.prevInstId) {
                let inst = this.annoInst.instances[instId - 1];
                this.prepareAnnoInst(inst);
                this.renderAnnoInst();
                this.prevInstId;
                if (inst) {
                    let areaM = Math.round(inst.area * this.scaleCoeff * this.scaleCoeff);
                    this.statisticsTextElem.innerHTML =
                    `class: ${this.labelsMap[inst.class].name}<br>
                    area: ${inst.area} px<br>
                    area: ${areaM} µm2<br>
                    objectId: ${inst.id}`;
                } else {
                    this.statisticsTextElem.innerHTML = `class: Background`;
                }
            }
        }
    }
}

function getClassLabel(data, w, h, labelsMap, x, y) {
    let n = data[(w * y + x) * 3];
    return labelsMap[n].name;
}

function getColorizeAnnoSem(data, w, h, labelsMap, alpha) {
    let anno = new Uint8ClampedArray(w * h * 4);
    for (i = 0; i < w * h; i++) {
        let v = data[3 * i];
        let color = labelsMap[v].color;
        anno[4 * i] = color[0];
        anno[4 * i + 1] = color[1];
        anno[4 * i + 2] = color[2];
        if (v > 0)
            anno[4 * i + 3] = alpha;
    }
    return anno;    
}

function colorizeAnnoInst(inst, labelsMap, alpha) {
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