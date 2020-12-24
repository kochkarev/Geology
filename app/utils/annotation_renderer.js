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

        this.createAnnoInstCache();
    }

    selectionChanged(xImage) {
        this.xImage = xImage;
        this.clearAnnoInstCache();
        this.renderImage(xImage, true);
        this.annoInst = null;
        this.prevInstId = null;
        this._annoInstClear();
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

    _annoInstClear() {
        const {w, h} = this.getSize();
        this.canvAnnoInst.width = w;
        this.canvAnnoInst.height = h;
        this.ctxAnnoInst.clearRect(0, 0, w, h);
    }

    getColorizedInst(inst, alpha) {
        if (inst == null) { // null or undefined
            return null;
        }
        if (!this.annoInstCache.get(inst.src).has(inst.id)) {
            this.annoInstCache.get(inst.src).set(inst.id, colorizeAnnoInst(inst, this.labelsMap, alpha));
        }
        return this.annoInstCache.get(inst.src).get(inst.id);
    }

    renderOneInst(inst, alpha=255) {
        // render anno to tmp canvas
        this.ctxAnnoInstTmp.clearRect(0, 0, this.xImage.w, this.xImage.h);
        let colorized = this.getColorizedInst(inst, alpha);
        if (colorized !== null) {
            this.ctxAnnoInstTmp.putImageData(new ImageData(colorized, inst.w, inst.h), inst.x, inst.y);
        }
        // render to anno-inst canvas
        const {w, h} = this.getSize();
        this._annoInstClear()
        this.ctxAnnoInst.drawImage(this.canvAnnoInstTmp, 0, 0, w, h);                   
    }

    resetPrevId() {
        this.annoInstPrevId = new Map();
        this.annoInstPrevId.set('GT', undefined);
        this.annoInstPrevId.set('PR', undefined);
    }

    createAnnoInstCache() {
        this.annoInstCache = new Map();
        this.annoInstCache.set('GT', new Map());
        this.annoInstCache.set('PR', new Map());
        this.resetPrevId();
    }

    clearAnnoInstCache() {
        this.annoInstCache.get('GT').clear();
        this.annoInstCache.get('PR').clear();
        this.resetPrevId();
    }

    getInst(src, x, y) {
        const xs = Math.round(x / this.scale);
        const ys = Math.round(y / this.scale);
        if (this.xImage?.annoInst.has(src)) {
            let instId = this.xImage.annoInst.get(src).instMap.data[(ys * this.xImage.w + xs) * 3];
            if (instId === 0) {
                return null;
            }
            return this.xImage.annoInst.get(src).instances[instId - 1];
        }
        return null;
    }

    renderAnnoInst(x, y) {
        if (this.renderOpts.inst === null || x == null || y == null) {
            this._annoInstClear();
        } else {
            if (this.renderOpts.inst === 'GT') {
                let inst = this.getInst('GT', x, y);
                if (inst == null && this.annoInstPrevId.get('GT') != null) {
                    this._annoInstClear();
                    this.annoInstPrevId.set('GT', null);
                } else if (inst != null && inst.id !== this.annoInstPrevId.get('GT')) {
                    this.renderOneInst(inst);
                    this.updateStatisticsInst(inst);
                    this.annoInstPrevId.set('GT', inst.id);
                }
            } else if (this.renderOpts.inst === 'PR') {

            }
        } 
    }

    changeScale(code, step=0.1) {
        if (code === 'BracketRight') {
            this.scale = Math.min(this.scale + step, 2);
        }
        else if (code === 'BracketLeft') {
            this.scale = Math.max(this.scale - step, 0.1);
        }
        this.resetPrevId();
        this.renderImage(this.xImage, false);
        this.renderAnnoSem();
        this.renderAnnoInst();
    }

    updateFromMain(xImage) {
        this.wXImages.update(xImage);
        if (this.xImage.id === xImage.id) {
            this.xImage = xImage;
            this.canvAnnoInstTmp.width = this.xImage.w;
            this.canvAnnoInstTmp.height = this.xImage.h;
        }
    }

    cursorMove(x, y) {
        if (!this.annoInst && this.xImage !== null) {
            this.updateStatisticsSem(this.xImage.annoSemanticGT, x, y);
        }
        this.renderAnnoInst(x, y);
    }

    updateStatisticsSem(annoSem, x, y) {
        const xs = Math.round(x / this.scale);
        const ys = Math.round(y / this.scale);
        let classLabel = getClassLabel(annoSem.data, annoSem.w, annoSem.h, this.labelsMap, xs, ys);
        if (classLabel !== null) {
            this.statisticsTextElem.innerHTML = `class: ${classLabel}`;
        } else {
            this.statisticsTextElem.innerHTML = ``;
        }
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
    const i = (w * y + x) * 3;
    if (i >= 0 && i < data.length) {
        let n = data[(w * y + x) * 3];
        return labelsMap[n].name;
    } else {
        return null;
    }
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