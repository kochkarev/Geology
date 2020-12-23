class XImageWrapper {
    constructor(xImage) {
        this.xImage = xImage;
        let li = document.createElement('li');
        li.className = "list-group-item";
        li.innerHTML = xImage.imageName;
        this.li = li;
    }
}

class XImageWrapperList {
    
    constructor(listGroupHTML, updateImageCallBack) {
        this.items = [];
        this.itemsMap = new Map();
        this.activeIdx = 0;
        this.activeIdxPrev = 0;
        this.listGroupHTML = listGroupHTML;
        this.updateImageCallBack = updateImageCallBack;
    }

    update(xImage) {
        if (this.itemsMap.has(xImage.id)) {
            let idx = this.itemsMap[xImage.id];
            this.items[idx].xImage = xImage;
        } else {
            let wrapper = new XImageWrapper(xImage);
            this.itemsMap[xImage.id] = this.items.length;
            this.items.push(wrapper);
            this.listGroupHTML.append(wrapper.li);
            if (this.items.length === 1) {
                this.selectByIndex(0);
            }
        }
    }

    updateMany(xImages) {
        for (const [key, imgStruct] of xImages.entries()) {
            this.update(imgStruct);
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
        this.updateImageCallBack(this.items[this.activeIdx].xImage);
    }
};

module.exports = {XImageWrapper: XImageWrapper, XImageWrapperList: XImageWrapperList};