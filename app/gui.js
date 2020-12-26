const {ipcRenderer} = require('electron');

const fs = require('fs');
const {AnnotationRenderer} = require('./utils/annotation_renderer');

const canvClassAnno = document.getElementById('canv-anno-sem');
const mainImageZone = document.getElementById('main-image-zone');

function getLabelsDecoded(config='./config/labels.json') {
    const jf = JSON.parse(fs.readFileSync(config));
    let labelsDecoded = {};
    for (const [label, className] of Object.entries(jf.LabelsToClasses)) {
        const color = jf.ClassesToColors[className];
        labelsDecoded[label] = {"name": className, "color": color};
    }
    return labelsDecoded;    
}


let R = new AnnotationRenderer(
    document.getElementById('files-list'),
    document.getElementById('img'),
    document.getElementById('canv-tmp-sem'),
    document.getElementById('canv-tmp-inst'),
    document.getElementById('canv-anno-sem'),
    document.getElementById('canv-anno-inst'),
    document.getElementById('statistics-text'),
    document.getElementById('selector-vis'),
    getLabelsDecoded(),
    scaleCoeff=0.4
);

mainImageZone.addEventListener('mousemove', (e) => {
    let rect = canvClassAnno.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    R.cursorMove(x, y);
});

document.addEventListener('keydown', (e) => {
    if (e.code === 'ArrowDown' || e.code === 'ArrowUp')
        R.wXImages.moveSelection(e.code);
    else if (e.code === 'BracketLeft' || e.code === 'BracketRight')
        R.changeScale(e.code);    
});

document.getElementById('btn_load').addEventListener('click', () => ipcRenderer.send('load-model'));
document.getElementById('btn_predict').addEventListener('click', () => ipcRenderer.send('predict'));

ipcRenderer.on('ximages-added', (e, items) => {
    R.wXImages.updateMany(items);
});

ipcRenderer.on('ximage-update', (e, ximage) => {
    R.updateFromMain(ximage);
});

ipcRenderer.on('model-loaded', () => {
    const loadBtn = document.getElementById('btn_load');
    loadBtn.disabled = true;
});

