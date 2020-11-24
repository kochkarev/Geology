const path = window.require('path');

class ShallowImgStruct {
    constructor(filePath) {
        this.filePath = filePath;
        this.fileName = path.parse(filePath).base;
        let li = document.createElement('li');
        li.className = "list-group-item";
        li.innerHTML = this.fileName;
        this.li = li;
    }
}

class ShallowImageList {
    
    constructor(listGroupHTML, updateImageCallBack) {
        this.items = [];
        this.activeIdx = 0;
        this.activeIdxPrev = 0;
        this.listGroupHTML = listGroupHTML;
        this.updateImageCallBack = updateImageCallBack;
    }

    addFile(filePath) {
        let imgStruct = new ShallowImgStruct(filePath);
        this.items.push(imgStruct);
        this.listGroupHTML.append(imgStruct.li);
        if (this.items.length === 1) {
            this.selectByIndex(0);
        }
    }

    addFiles(filePaths) {
        for (let filePath of filePaths) {
            this.addFile(filePath);
        }
    }

    moveSelection(code) {
        let n = this.items.length;
        if (n === 0)
            return;
        let idx = 0;
        if (code === 'ArrowDown')
            idx = (this.activeIdx + 1) % n;
        if (code === 'ArrowUp')
            idx = (n + this.activeIdx - 1) % n;
        this.selectByIndex(idx);
    }

    selectClick(e) {
        if (this.items.length == 0)
            return;
        let li = e.target.closest('li');
        let idx = Array.from(this.listGroupHTML.children).indexOf(li);
        this.selectByIndex(idx);
    }

    selectByIndex(newIndex) {
        this.activeIdxPrev = this.activeIdx;
        this.activeIdx = newIndex;
        this.listGroupHTML.children[this.activeIdxPrev].classList.remove('active');
        this.listGroupHTML.children[this.activeIdx].classList.add('active');
        this.updateImageCallBack(this.items[this.activeIdx], this.activeIdx);
    }
};

module.exports = {ShallowImgStruct: ShallowImgStruct, ShallowImageList: ShallowImageList};