const fs = window.require('fs');

const {XImageWrapperList} = require('./structs_ui');


class Context {
    constructor(labelsMap) {
        this.instCache = new Map();
        this.instPrevId = new Map();
        this.labelsMap = labelsMap;
    }

    resetInstCache() {
        for (const [k, v] of this.instCache.entries()) {
            v.clear();
        }
        this.resetPrevInstId();
    }

    resetPrevInstId() {
        this.instPrevId.clear();
    }

    instColorized(inst, alpha) {
        if (inst == null) {
            return null;
        }
        if (!this.instCache.has(inst.src)) {
            this.instCache.set(inst.src, new Map());
        }
        if (!this.instCache.get(inst.src).has(inst.id)) {
            this.instCache.get(inst.src).set(inst.id, colorizeAnnoInst(inst, this.labelsMap, alpha));
        }
        return this.instCache.get(inst.src).get(inst.id);
    }

    setPrevInstId(src, id) {
        this.instPrevId.set(src, id);
    } 

    getPrevInstId(src) {
        return this.instPrevId.get(src);
    }

}

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
        this.scale = 1;

        this.wXImages = new XImageWrapperList(filesList, (x) => this.selectionChanged(x));

        this.renderOpts = {
            sem: null,
            inst: null
        }

        this.visSelector.addEventListener('change', (e) => this.visSelectorChanged(e.target.value));
        this.ctx = new Context(this.labelsMap);
    }

    visSelectorChanged(val) {
        switch (val) {
            case 'SRC':
                this.renderOpts.sem = null;
                this.renderOpts.inst = null;
                break;
            case 'GT':
                this.renderOpts.sem = 'GT';
                this.renderOpts.inst = 'GT';
                break;
            case 'PR':
                this.renderOpts.sem = 'PR';
                this.renderOpts.inst = 'PR';
                break;
            case 'ERR':
                this.renderOpts.sem = 'ERR';
                this.renderOpts.inst = null;
                break;
        }
        this.renderAnnoSem();
        this.clearAnnoInst();
        this.resetStatistics();
    }

    selectionChanged(xImage) {
        this.xImage = xImage;
        this.ctx.resetInstCache();
        this.renderImage(xImage, true);
        this.ctx.resetPrevInstId();
        this.clearAnnoInst();
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

    loadAnnoSem(anno, alpha=150) {
        let annoSemColorized = getColorizeAnnoSem(anno.data, anno.w, anno.h, this.labelsMap, alpha=alpha);
        this.canvAnnoSemTmp.width = anno.w;
        this.canvAnnoSemTmp.height = anno.h;
        this.ctxAnnoSemTmp.putImageData(new ImageData(annoSemColorized, anno.w, anno.h), 0, 0);
    }

    clearAnnoSem() {
        const {w, h} = this.getSize();
        this.canvAnnoSem.width = w;
        this.canvAnnoSem.height = h;
        this.ctxAnnoSem.clearRect(0, 0, w, h);
    }

    updateAnnoSem() {
        const {w, h} = this.getSize();
        this.canvAnnoSem.width = w;
        this.canvAnnoSem.height = h;
        this.ctxAnnoSem.clearRect(0, 0, w, h);
        this.ctxAnnoSem.drawImage(this.canvAnnoSemTmp, 0, 0, w, h);
    }

    renderAnnoSem() {
        if (this.renderOpts.sem === null) {
            this.clearAnnoSem();
        } else {
            let annoSem = this.xImage.annoSemantic.get(this.renderOpts.sem);
            if (annoSem == null) {
                this.clearAnnoSem();
                return;
            }
            if (this.annoSemCached !== annoSem) {
                this.loadAnnoSem(annoSem);
                this.annoSemCached = annoSem;
            }                
            this.updateAnnoSem();
        }
    }

    clearAnnoInst() {
        const {w, h} = this.getSize();
        this.canvAnnoInst.width = w;
        this.canvAnnoInst.height = h;
        this.ctxAnnoInst.clearRect(0, 0, w, h);
    }

    renderOneInst(inst, alpha=255) {
        // render anno to tmp canvas
        this.ctxAnnoInstTmp.clearRect(0, 0, this.xImage.w, this.xImage.h);
        let colorized = this.ctx.instColorized(inst, alpha);
        if (colorized !== null) {
            this.ctxAnnoInstTmp.putImageData(new ImageData(colorized, inst.w, inst.h), inst.x, inst.y);
        }
        // render to anno-inst canvas
        const {w, h} = this.getSize();
        this.clearAnnoInst()
        this.ctxAnnoInst.drawImage(this.canvAnnoInstTmp, 0, 0, w, h);                   
    }

    getInst(src, x, y) {
        const xs = Math.round(x / this.scale);
        const ys = Math.round(y / this.scale);
        if (this.xImage?.annoInst.has(src)) {
            let instMap = this.xImage.annoInst.get(src).instMap;
            if (instMap == null) {
                return null;
            }
            let v1 = instMap.data[(ys * this.xImage.w + xs) * 3];
            let v2 = instMap.data[(ys * this.xImage.w + xs) * 3 + 1];
            let v3 = instMap.data[(ys * this.xImage.w + xs) * 3 + 2];
            let instId = v1 * 256 * 256 + v2 * 256 + v3;
            if (instId === 0) {
                return null;
            }
            return this.xImage.annoInst.get(src).instances[instId - 1];
        }
        return null;
    }

    renderAnnoInst(x, y) {
        let src = this.renderOpts.inst;
        let inst = this.getInst(src, x, y);
        if (inst == null && this.ctx.getPrevInstId(src) != null) {
            this.clearAnnoInst();
            this.ctx.setPrevInstId(src, null);
        } else if (inst != null && inst.id !== this.ctx.getPrevInstId(src)) {
            this.renderOneInst(inst);
            this.ctx.setPrevInstId(src, inst.id);
        }
        this.updateStatisticsInst(inst);
    }

    changeScale(code, step=0.1) {
        if (code === 'BracketRight') {
            this.scale = Math.min(this.scale + step, 2);
        }
        else if (code === 'BracketLeft') {
            this.scale = Math.max(this.scale - step, 0.1);
        }
        this.ctx.resetPrevInstId();
        this.renderImage(this.xImage, false);
        this.renderAnnoSem();
        this.clearAnnoInst();
    }

    updateFromMain(xImage) {
        this.wXImages.update(xImage);
        if (this.xImage.id === xImage.id) {
            this.xImage = xImage;
            this.canvAnnoInstTmp.width = this.xImage.w;
            this.canvAnnoInstTmp.height = this.xImage.h;
            this.renderAnnoSem();
        }
    }

    cursorMove(x, y) {
        if (this.xImage == null)
            return;
        let semSrc = this.renderOpts.sem;
        let instSrc = this.renderOpts.inst;
        if (semSrc !== null && this.xImage.annoSemantic.has(semSrc) && (instSrc === null || this.xImage.annoInst.get(instSrc) == null)) {
            this.updateStatisticsSem(this.xImage.annoSemantic.get(semSrc), x, y);
        }
        if (instSrc !== null && this.xImage.annoInst.get(instSrc) != null) {
            this.renderAnnoInst(x, y);
        }
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

    resetStatistics() {
        this.statisticsTextElem.innerHTML = ``;
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