const path = require('path');
const fs = require('fs');


class InstAnnotation {
	constructor() {
		this.instMap = null;
		this.instances = [];
	}

	addInst(inst) {
		this.instances.push(inst);
	}

	updateInstMap(instMap) {
		this.instMap = instMap;
	}

	getInstByCoords(x, y) {
		if (!this.instMap) {
			return null;
		}
		let inst_id = this.instMap.data[y * this.instMap.h + x];
		return this.instances[inst_id];
	}
}

class ImageList {
    
    constructor(backend, renderer) {
        this.backend = backend
        this.renderer = renderer
        this.items = [];
        this.activeIdx = 0;
        this.activeIdxPrev = 0;
    }

    addImage(fullFilePath) {
        let imageItem = {
			id: this.items.length,
            imageFullPath: fullFilePath,
            imageName: path.parse(fullFilePath).base,
			annotationFullPath: this.readAnnotation(fullFilePath),
			annotation: new InstAnnotation()
		};
        this.items.push(imageItem);
    }

    addImages(fullFilePaths) {
        for (let fullFilePath of fullFilePaths) {
            this.addImage(fullFilePath)
        }
        console.log(this.items);
    }

    updateRendererImage() {
        //this.renderer.send();
	}
	
	readAnnotation(imageFullPath) {
		let p = path.parse(imageFullPath);
		let maskFullPath = path.join(p.dir, p.name + '_mask' + '.png');
		try{
			return fs.existsSync(maskFullPath) ? maskFullPath : null;
		} catch (e) {
			return null;
		}
	}

	getActiveItem() {
		return this.items.length === 0 ? null : this.items[this.activeIdx]
	}

	changeActiveImageIdx(newActiveIdx = 0) {
		this.activeIdxPrev = this.activeIdx;
		this.activeIdx = newActiveIdx;
		let imgStruct = this.items[newActiveIdx];
		console.log(imgStruct);
		this.backend.getFullAnnotation(imgStruct.annotationFullPath, imgStruct.id);
	}

	updateAnnoInst(inst) {
		this.items[inst.imgid].annotation.addInst(inst);
	}

	updateAnnoInstMap(instMap) {
		this.items[instMap.imgid].annotation.updateInstMap(instMap);
	}
}


module.exports = {InstdAnnotation: InstAnnotation, ImageList: ImageList}