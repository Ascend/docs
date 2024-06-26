$(document).ready(function () {
    $.reset_selection = function (elem) {
        elem.parent().children().each(function () {
            $(this).removeClass("selected");
        });
    }

    $.get_options = function () {
        var options = {};
        $('#col-values').children().each(function () {
            var elem = $(this).find(".selected").each(function () {
                var id = $(this).attr("id").split("-");
                var category = id[0];
                var value = id[1];
                if(category == 'cann' || category == 'pytorch_npu')
                    options[category] = $(this).text();
                else
                    options[category] = value;
            });
        });
        return options;
    }

    $.update_table = function () {
        var options = $.get_options();
        var pytorch_version = options['pytorch'];
        var match_versions = pytorch_versions[pytorch_version];
        $("#pytorch_npu-version").text(match_versions['torch_npu']);
        $("#cann-version").text(match_versions['cann']);
    }

    $("#col-values").on("click", ".values-element", function () {
        id = $(this).attr("id");
        fields = id.split("-");
        if (fields[0] == "pytorch_npu" || fields[0] == "cann")
            return;

        $.reset_selection($(this));
        $(this).addClass("selected");
        $.update_table();
        $.gen_content();
    });

    
    $.gen_content = function () {
        var options = $.get_options();
        var pytorch_version = options['pytorch'];
        var match_versions = pytorch_versions[pytorch_version];
        if (options['install_type'] == "docker") {
            var dockerCommand = `
docker run \\
    --name cann_container \\
    --device /dev/davinci1 \\
    --device /dev/davinci_manager \\
    --device /dev/devmm_svm \\
    --device /dev/hisi_hdc \\
    -v /usr/local/dcmi:/usr/local/dcmi \\
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \\
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \\
    -v /etc/ascend_install.info:/etc/ascend_install.info \\
    -e DRIVER_PATH=/usr/local/Ascend/driver \\
    -it ${match_versions['docker']} bash
             `;
            
            $('#codecell0').html(dockerCommand);
            $('#install-pytorch-source-section').hide();
            $('#install-pytorch-pip-section').hide();
            $('#install-pytorch-docker-section').show();
        } else if (options['install_type'] == "pip") {
            $('#codecell1').html("# install torch<br>");
            if(options['arch'] == "aarch64")
                $('#codecell1').append("pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==" + options['pytorch']);
            else
                $('#codecell1').append("pip3 install torch=="+options['pytorch']+"+cpu  --index-url https://download.pytorch.org/whl/cpu");

            $("#codecell1").append("<br><br># install torch-npu<br>pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-npu==" + options['pytorch_npu']);

            $('#install-pytorch-source-section').hide();
            $('#install-pytorch-docker-section').hide();
            $('#install-pytorch-pip-section').show();
        } else {
            $("#codecell4").html("# install requirements<br>conda install cmake ninja git<br><br># get torch source<br>git clone -b v"+options['pytorch']+" --recursive https://github.com/pytorch/pytorch<br>cd pytorch<br>git submodule sync<br>git submodule update --init --recursive<br><br># install torch<br>pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt<br>export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-\"$(dirname $(which conda))/../\"}<br>USE_CUDA=0 python setup.py develop");

            $('#codecell4').append("<br><br># get torch-npu source<br>git clone https://github.com/ascend/pytorch.git -b "+match_versions['npu_branch']+" --depth 1 pytorch_npu<br>cd pytorch_npu<br><br># install torch-npu<br>pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt<br>bash ci/build.sh --python=$(python --version 2>&1 | awk '{print $2}' | cut -d '.' -f 1,2)<br>pip install dist/torch_npu*.whl");

            $('#install-pytorch-pip-section').hide();
            $('#install-pytorch-docker-section').hide();
            $('#install-pytorch-source-section').show();
        }
       
    }

    $.update_table();
    $.gen_content();
});
