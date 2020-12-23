const fs = window.require('fs');

const {XImageWrapperList} = require('./structs_ui');


class AnnotationRenderer {

    constructor(filesList, img, canvAnnoSemTmp, canvAnnoInstTmp, canvAnnoSem, canvAnnoInst, statisticsTextElem, visSelector, labelsMap, scaleCoeff) {
        this.imgElem = img;
        this.canvAnnoSemTmp = canvAnnoSemTmp;
        this.canvAnnoInstTmp = canvAnnoInstTmp;
        this.canvAnnoSem = canvAnnoSem;
        this.canvAnnoInst = canvAnnoInst;
        this.statisticsTextElem = statisticsTextElem;
        this.visSelector = visSelector;

        this.labelsMap = labelsMap;
        this.scaleCoeff = scaleCoeff;

        this.ctxAnnoSemTmp = this.canvAnnoSemTmp.getContext('2d');
        this.ctxAnnoInstTmp = this.canvAnnoInstTmp.getContext('2d');
        this.ctxAnnoSem = this.canvAnnoSem.getContext('2d');
        this.ctxAnnoInst = this.canvAnnoInst.getContext('2d');
        
        this.xImage = null;
        this.annoInst = null;
        this.prevInstId = null;
        this.scale = 1;

        this.instColorized = new Map();

        this.wXImages = new XImageWrapperList(filesList, (x) => this.selectionChanged(x));

        this.renderOpts = {
            sem: null,
            inst: 'GT'
        }

        this.visSelector.addEventListener('change', (e) => {
            switch (e.target.value) {
                case 'SRC':
                    this.renderOpts.sem = null;
                    break;
                case 'GT':
                    this.renderOpts.sem = 'GT';
                    break;
                case 'PR':
                    this.renderOpts.sem = null;
                    break;
                case 'ERR':
                    this.renderOpts.sem = null;
                    break;
            }
            this.renderAnnoSem();
        })

    }

    selectionChanged(xImage) {
        this.xImage = xImage;
        this.renderImage(xImage, true);
        this.annoInst = null;
        this.prevInstId = null;
        this.clearAnnoInst();
        this.instColorized.clear();
        ipcRenderer.send('active-image-update', xImage.id);
        this.renderAnnoSem();
    }

    getSize() {
        return {
            w: Math.round(this.scale * this.xImage.w),
            h: Math.round(this.scale * this.xImage.h)
        };
    }

    renderImage(xImage, full) {
        if (full) {
            const {w, h} = this.getSize();
            let _img = fs.readFileSync(xImage.imagePath).toString('base64');
            this.imgElem.src = `data:image/jpg;base64,${_img}`;
            this.imgElem.onload = () => {
                this.imgElem.width = w;
                this.imgElem.height = h;
            };
        } else {
            const {w, h} = this.getSize();
            this.imgElem.width = w;
            this.imgElem.height = h;
        }
    }

    _annoSemLoad(anno, alpha=150) {
        // transform mask to colorized annotation
        let annoSemColorized = getColorizeAnnoSem(anno.data, anno.w, anno.h, this.labelsMap, alpha=alpha);
        // render to temporary canvas
        this.canvAnnoSemTmp.width = anno.w;
        this.canvAnnoSemTmp.height = anno.h;
        this.ctxAnnoSemTmp.putImageData(new ImageData(annoSemColorized, anno.w, anno.h), 0, 0);
    }

    _annoSemClear() {
        const {w, h} = this.getSize();
        this.canvAnnoSem.width = w;
        this.canvAnnoSem.height = h;
        this.ctxAnnoSem.clearRect(0, 0, w, h);
    }

    _annoSemUpdate() {
        const {w, h} = this.getSize();
        this.canvAnnoSem.width = w;
        this.canvAnnoSem.height = h;
        this.ctxAnnoSem.clearRect(0, 0, w, h);
        this.ctxAnnoSem.drawImage(this.canvAnnoSemTmp, 0, 0, w, h);
    }

    renderAnnoSem() {
        if (this.renderOpts.sem === null) {
            this._annoSemClear();
        } else if (this.renderOpts.sem === 'GT') {
            if (this.annoSemCached === this.xImage.annoSemanticGT) {
                this._annoSemUpdate();
            } else {
                this._annoSemLoad(this.xImage.annoSemanticGT);
                this._annoSemUpdate();
                this.annoSemCached = this.xImage.annoSemanticGT;
            }
        }
    }

    clearAnnoInst() {
        const {w, h} = this.getSize();
        this.canvAnnoInst.width = w;
        this.canvAnnoInst.height = h;
        this.ctxAnnoInst.clearRect(0, 0, w, h);
    }

    renderAnnoInst(inst, alpha=255) {
        // render anno to tmp canvas
        this.ctxAnnoInstTmp.clearRect(0, 0, this.xImage.w, this.xImage.h);
        if (inst) {
            if (!this.instColorized.has(inst.id)) {
                this.instColorized.set(inst.id, colorizeAnnoInst(inst, this.labelsMap, alpha));
            }
            let colorized = this.instColorized.get(inst.id);
            this.ctxAnnoInstTmp.putImageData(new ImageData(colorized, inst.w, inst.h), inst.x, inst.y);
        }

        // render to anno-inst canvas
        const {w, h} = this.getSize();
        this.clearAnnoInst()
        this.ctxAnnoInst.drawImage(this.canvAnnoInstTmp, 0, 0, w, h);                   
    }

    changeScale(code, step=0.1) {
        if (code === 'BracketRight') {
            this.scale = Math.min(this.scale + step, 2);
        }
        else if (code === 'BracketLeft') {
            this.scale = Math.max(this.scale - step, 0.1);
        }
        this.renderImage(this.xImage, false);
        this.renderAnnoSem();
        //this.renderAnnoInst();
    }

    updateFromMain(xImage) {
        this.wXImages.update(xImage);
        let anno = xImage.annoInstGT;
        if (anno.imgId !== this.xImage.id)
            return;
        this.annoInst = anno;
        this.canvAnnoInstTmp.width = this.xImage.w;
        this.canvAnnoInstTmp.height = this.xImage.h;
    }

    cursorMove(x, y) {
        if (!this.annoInst && this.xImage !== null) {
            this.updateStatisticsSem(this.xImage.annoSemanticGT, x, y);
        }

        if (this.annoInst?.instMap) {
            const xs = Math.round(x / this.scale);
            const ys = Math.round(y / this.scale);

            let instId = this.annoInst.instMap.data[(ys * this.annoInst.instMap.w + xs) * 3];
            if (instId !== this.prevInstId) {
                let inst = this.annoInst.instances[instId - 1];
                this.renderAnnoInst(inst);
                this.prevInstId;
                this.updateStatisticsInst(inst);
            }
        }
    }

    updateStatisticsSem(annoSem, x, y) {
        const xs = Math.round(x / this.scale);
        const ys = Math.round(y / this.scale);
        let classLabel = getClassLabel(annoSem.data, annoSem.w, annoSem.h, this.labelsMap, xs, ys);
        this.statisticsTextElem.innerHTML = `class: ${classLabel}`;
    }

    updateStatisticsInst(inst) {
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


module.exports = {AnnotationRenderer: AnnotationRenderer};