const {ipcRenderer} = require('electron');

const fs = require('fs');
const {XImageWrapperList} = require('./utils/structs_ui');
const {ActiveImageWithAnnotationRenderer} = require('./utils/annotation_renderer');

const canvClassAnno = document.getElementById('canv-anno-sem');
const mainImageZone = document.getElementById('main-image-zone');
const visSelector = document.getElementById('selector-vis');


function getLabelsDecoded(config='./config/labels.json') {
    const jf = JSON.parse(fs.readFileSync(config));
    let labelsDecoded = {};
    for (const [label, className] of Object.entries(jf.LabelsToClasses)) {
        const color = jf.ClassesToColors[className];
        labelsDecoded[label] = {"name": className, "color": color};
    }
    return labelsDecoded;    
}


let R = new ActiveImageWithAnnotationRenderer(
    document.getElementById('img'),
    document.getElementById('canv-tmp-sem'),
    document.getElementById('canv-tmp-inst'),
    document.getElementById('canv-anno-sem'),
    document.getElementById('canv-anno-inst'),
    document.getElementById('statistics-text'),
    getLabelsDecoded(),
    scaleCoeff=0.4
);


mainImageZone.addEventListener('mousemove', (e) => {
    let rect = canvClassAnno.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    R.cursorMove(x, y);
});

visSelector.addEventListener('change', (e) => {
    console.log(e.target.value);
});

let imgList = new XImageWrapperList(
    document.getElementById('files-list'),
    (xImage) => R.changeContext(xImage)
);

document.addEventListener('keydown', (e) => {
    if (e.code === 'ArrowDown' || e.code === 'ArrowUp')
        imgList.moveSelection(e.code);
    else if (e.code === 'BracketLeft' || e.code === 'BracketRight')
        R.changeScale(e.code);    
});

imgList.listGroupHTML.addEventListener('click', e => imgList.selectClick(e));

document.getElementById('btn_load').addEventListener('click', () => ipcRenderer.send('load-model'));
document.getElementById('btn_predict').addEventListener('click', () => ipcRenderer.send('predict'));
// document.getElementById('btn_stop').addEventListener('click', () => ipcRenderer.send('stop-algo'));

ipcRenderer.on('images-added', (e, items) => {
    imgList.updateMany(items);
});

ipcRenderer.on('anno-loaded', (e, anno) => {
    R.annoInstUpdateFromMain(anno);
});

