const { app, BrowserWindow, ipcMain, Menu, dialog } = require('electron');
const { format } = require('url');
const path = require('path');
const { BackendCommunicator } = require('./utils/backend');
const { XImageCollection } = require('./utils/structs');

const backendCfg = {
	pyEnvName: 'tf',
	pyEnvPath: 'C:/Users/xubiker/Anaconda3/envs/tf/python',
	srcPath: './backend/server.py',
	modelName: 'model_46_0.07'
};

let backend = null;
let xImages = null;


function handlerArray(arr, header) {
	if ('ext' in header) {
		if (header.ext == 'inst') {
			let inst = {
				'src': header.src, 'id': header.id, 'x': header.x, 'y': header.y, 'w': header.shape[1], 'h': header.shape[0],
				'class': header.class, 'imgid': header.imgid, 'mask': arr, 'area': header.area
			};
			xImages.updateAnnoInst(inst);
		} else if (header.ext == 'inst-map') {
			let instMap = {'w': header.shape[1], 'h': header.shape[0], 'imgid': header.imgid, 'data': arr, 'src': header.src};
			xImages.updateAnnoInstMap(instMap);
		} else if (header.ext == 'pred') {
			let prediction = {'src': 'PR', 'id': header.imgid, 'w': header.shape[1], 'h': header.shape[0], 'data': arr};
			xImages.updateAnnoSem(prediction);
		} else if (header.ext == 'err') {
			let errorMap = {'src': 'ERR', 'id': header.imgid, 'w': header.shape[1], 'h': header.shape[0], 'data': arr};
			xImages.updateAnnoSem(errorMap);
		}
	} else {
		console.log(`#arr: shape[${header.shape}]. Got ${arr.length} bytes`);
	}
}

function handlerSignal(s) {
	if (s.startsWith('A')) {
		let imgId = parseInt(s.slice(1));
		xImages.onAnnotationLoaded(imgId, 'GT');
	} else if (s.startsWith('B')) {
		let imgId = parseInt(s.slice(1));
		xImages.onAnnotationLoaded(imgId, 'PR');
	} else if (s.startsWith('L')) {
		if (parseInt(s.slice(1)) === 1) {
			xImages.onModelLoaded();
		}
	}
}

function handlerString(s) {
	console.log('#str: ' + s);
}

app.on('ready', () => {
	win = new BrowserWindow({ width: 1400, height: 800, webPreferences: { nodeIntegration: true, enableRemoteModule: false}})
	win.loadURL(format({
		pathname: path.join(__dirname, 'index.html'),
		protocol: 'file:',
		slashes: true,
	}))

	win.webContents.on('did-finish-load', () => {
        backend = new BackendCommunicator(
			backendCfg,
			(arr, header) => handlerArray(arr, header),
			(sig) => handlerSignal(sig),
			(str) => handlerString(str)
		);
		xImages = new XImageCollection(backend, win.webContents);
	});

	ipcMain.on('active-image-update', (event, args) => xImages.onActiveImageUpdate(args));
	ipcMain.on('load-model', (event, args) => backend.loadModel());
	ipcMain.on('predict', (event, args) => xImages.predict())

	const mainMenu = Menu.buildFromTemplate(menuTemplate);
	Menu.setApplicationMenu(mainMenu);

	win.webContents.openDevTools();    	
	
})

app.on('before-quit', () => {
	backend.stop();
});

const menuTemplate = [
	{
		label: 'Project',
		submenu: [
		{
			label: 'Open'
		},
		{
			label: 'Import images',
			accelerator: 'CmdOrCtrl+O',
			click: () => {dialog_import_files()}
		},
		{
			label: 'Save'
		},
		{
			type: 'separator'
		},
		{
			label: 'Quit',
			accelerator: process.platform == 'darwin' ? 'Command+Q' : 'Ctrl+Q',
			click: () => {app.quit()}
		}
		],
	},
	{
		label: 'Edit',
		submenu: [
			
		]
	},
	{
		label: 'Help'
	}
]


function dialog_import_files() {
	dialog.showOpenDialog({
		title: 'Select images to import...',
		properties: ['openFile', 'multiSelections'],
		defaultPath: __dirname + "\\sample_data",
		buttonLabel: "Select",
		filters: [
		    { name: 'Images', extensions: ['jpg']}
		]
		})
		.then(res => {
			if (!res.canceled) {
				xImages.addFromPaths(res.filePaths);
				win.webContents.send('ximages-added', Array.from(xImages.items.values()));
			}
		})
		.catch( (e) =>
			console.log('promise rejected: ' + e)
		);
}