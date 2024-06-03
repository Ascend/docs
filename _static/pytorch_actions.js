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
                if(category == 'pytorch')
                    options[category] = value;
                else
                    options[category] = $(this).text();
            });
        });
        return options;
    }

    $.update_table = function () {
        var options = $.get_options();
        var pytorch_version = options['pytorch'];
        match_versions = pytorch_versions[pytorch_version];
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
        if (options['install_type'] == "pip") {
            if(options['arch'] == "aarch64")
                $('#install-pytorch-pip').html("pip3 install torch==" + options['pytorch']);
            else
                $('#install-pytorch-pip').html("pip3 install torch=="+options['pytorch']+"+cpu  --index-url https://download.pytorch.org/whl/cpu");

            $("#install-pytorch_npu-pip").html("pip3 install torch-npu==" + options['pytorch_npu']);

            $('#install-pytorch-source-section').hide();
            $('#install-pytorch_npu-source-section').hide();
            $('#install-pytorch-pip-section').show();
            $('#install-pytorch_npu-pip-section').show();
        } else {
            $("#install-pytorch-source-build").html("# install requirements<br>conda install cmake ninja<br><br># get source<br>git clone -b "+options['pytorch']+" --recursive https://github.com/pytorch/pytorch<br>cd pytorch<br>git submodule update --init --recursive<br><br># install PyTorch<br>pip install -r requirements.txt<br>export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-\"$(dirname $(which conda))/../\"}<br>python setup.py develop");

            $('#install-pytorch_npu-source-build').html("# get source<br>git clone https://github.com/ascend/pytorch.git -b "+options['pytorch_npu']+" --depth 1 pytorch_npu<br>cd pytorch_npu<br><br>#install pytorch_npu<br>bash ci/build.sh --python=$(python --version 2>&1 | awk '{print $2}' | cut -d '.' -f 1,2)");

            $('#install-pytorch-pip-section').hide();
            $('#install-pytorch_npu-pip-section').hide();
            $('#install-pytorch-source-section').show();
            $('#install-pytorch_npu-source-section').show();
        }
       
    }

    $.update_table();
    $.gen_content();
});
