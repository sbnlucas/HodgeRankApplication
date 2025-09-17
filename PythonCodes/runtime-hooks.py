# runtime-hooks.py
import sys
import os
import platform

def set_runtime_paths():
    """Configura paths de runtime para diferentes sistemas operacionais"""
    
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        
        # Configura paths para pythonnet
        pythonnet_path = os.path.join(base_path, 'pythonnet')
        if platform.system() == 'Windows':
            runtime_path = os.path.join(pythonnet_path, 'runtime')
        else:
            runtime_path = os.path.join(pythonnet_path, 'lib')
        
        if os.path.exists(runtime_path):
            os.environ['PATH'] = runtime_path + os.pathsep + os.environ['PATH']
        
        # Configura paths para .NET
        dotnet_path = os.path.join(base_path, 'dotnet')
        if os.path.exists(dotnet_path):
            os.environ['PATH'] = dotnet_path + os.pathsep + os.environ['PATH']
            os.environ['DOTNET_ROOT'] = dotnet_path
            
            # Configurações específicas para Linux/macOS
            if platform.system() != 'Windows':
                lib_path = os.path.join(dotnet_path, 'lib')
                if os.path.exists(lib_path):
                    if 'LD_LIBRARY_PATH' in os.environ:
                        os.environ['LD_LIBRARY_PATH'] = lib_path + os.pathsep + os.environ['LD_LIBRARY_PATH']
                    else:
                        os.environ['LD_LIBRARY_PATH'] = lib_path
                    
                    if 'DYLD_LIBRARY_PATH' in os.environ and platform.system() == 'Darwin':
                        os.environ['DYLD_LIBRARY_PATH'] = lib_path + os.pathsep + os.environ['DYLD_LIBRARY_PATH']

# Executa a configuração quando o módulo é importado
set_runtime_paths()