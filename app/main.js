const { app, BrowserWindow, ipcMain, Menu, dialog } = require('electron');
const { format } = require('url');
const path = require('path');
const { spawn } = require('child_process');


const python_env_name = 'tf'
const python_env_path = 'C:/Users/xubiker/Anaconda3/envs/tf/python'
const python_backend_path = './backend/server.py'

let win = null;


class Backend {

	constructor(win) {
		this.win = win;
		this.state = null;
		this.hist = null;
		this.proc = spawn(`activate ${python_env_name} && ${python_env_path} ${python_backend_path}`, {
			// detached: false,
			shell: true,
		});
		this.proc.stdout.on('data', (buf) => {
			while (true) {
				let i = buf.indexOf('\n');
				if (i === -1) {
					this.hist = !this.hist ? buf : Buffer.concat([this.hist, buf]);
					break;
				} else {
					this.hist = !this.hist ? buf.slice(0, i - 1) : Buffer.concat([this.hist, buf.slice(0, i - 1)]);
					this.process_backend_message(this.hist);
					this.hist = null;
					buf = buf.slice(i + 1);
				}
			}
		});
		this.proc.stdout.on('end', () => {console.log('#end')});
		this.proc.stderr.on('data', (data) => console.error(`stderr: ${data}`));
		this.proc.on('exit', (code, signal) => console.log(`backend process exited with code ${code} and signal ${signal}`));
	}

	load_CNN() {
		this.proc.stdin.write(JSON.stringify({'type': 'load-CNN'}) + '\n');
	}

	stop_backend() {
		this.proc.stdin.write(JSON.stringify({'type': 'shutdown'}) + '\n');
		console.log('closing backend');
		this.proc.kill();
	}

	ping() {
		this.proc.stdin.write(JSON.stringify({'type': 'ping'}) + '\n');
	}

	ping_image() {
		this.proc.stdin.write(JSON.stringify({'type': 'ping_image'}) + '\n');
    }
    
    ping_coord(x, y) {
		this.proc.stdin.write(JSON.stringify({'type': 'ping_coord', 'x': x, 'y': y}) + '\n');
    }

	process_backend_message(message) {
		if (this.state === null) {
			try {
				let message_json = JSON.parse(message.toString());
				switch (message_json['type']) {
					case 'string':
						console.log('#backend-string: ' + message_json['content']);
						break;
					case 'image':
						console.log(`#backend-image: ${message_json['width']}x${message_json['height']}. Waiting for data`);
						this.state = 'image';
						break;
				}
			} catch (e) {
				console.log(`unsupported message from backend: ${message.toString()}`)
			}
		} else if (this.state === 'image') {
			this.process_image(new Uint8Array(message));
			this.state = null;
		}
	}

	process_image(arr) {
		console.log(`#backend-array: ${arr.length}`);
		this.win.webContents.send('anno-update', arr);
	}
	
	send_annotation(header, data) {
		this.proc.stdin.write(`${JSON.stringify(header)}\n`);
		this.proc.stdin.write(data);
	}

	stop_algo() {
	}

}

let backend = null;


function foo(x, y) {
    if (backend !== null) {
        backend.ping_coord(x, y);
    }
}

app.on('ready', () => {
	win = new BrowserWindow({ width: 1400, height: 800, webPreferences: { nodeIntegration: true, enableRemoteModule: false}})
	win.loadURL(format({
		pathname: path.join(__dirname, 'index.html'),
		protocol: 'file:',
		slashes: true,
	}))

	win.webContents.on('did-finish-load', () => {
		backend = new Backend(win);
		backend.load_CNN();
	});

	ipcMain.on('stop-algo', (event, arg) => backend.stop_algo());
	// ipcMain.on('btn_req_click', (event, arg) => backend.ping_image());
    ipcMain.on('anno', (event, args) => backend.send_annotation(args[0], args[1]));
    ipcMain.on('mouse-move', (event, args) => foo(args[0], args[1]));

	const mainMenu = Menu.buildFromTemplate(menuTemplate);
	Menu.setApplicationMenu(mainMenu);

    //win.webContents.openDevTools();    
})

app.on('before-quit', () => {
	backend.stop_backend();
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
			console.log(res);
			if (!res.canceled) {
				win.webContents.send('files-added', JSON.stringify(res.filePaths));
			}
		})
		.catch( () =>
			console.log('promise rejected')
		);
}