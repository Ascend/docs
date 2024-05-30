$(document).ready(function () {
    let cann_version_select = $('#cann-version');
    $.each(package_info, function (key, value) {
        cann_version_select.append(new Option("CANN: " + key, key));
    });

    $.reset_selection = function (elem) {
        elem.parent().children().each(function () {
            $(this).removeClass("selected");
        });
    }

    $("#col-values").on("click", ".values-element", function () {
        fields = $(this).attr("id").split("-");
        if (fields[1] == "version")
            return;
        $.reset_selection($(this));
        $(this).addClass("selected");
        $.gen_content($(this));
    });

    $("#col-values").on("change", "select", function () {
        // select cann, driver, formware versions.
        $.reset_selection($(this));
        $('#driver-version').text("Driver");
        $('#firmware-version').text("Firmware");

        if ($(this).val() != "na") {
            $(this).addClass("selected");
            $('#driver-version').addClass("selected");
            $('#firmware-version').addClass("selected");

            var npu_item = $('#row-npu').find(".selected");
            var npu_type = npu_item.attr("id").split("-")[1];
            var cann_version = $(this).val();
            var driver_version = package_info[cann_version][npu_type].driver_version;
            var firmware_version = package_info[cann_version][npu_type].firmware_version;
            $('#driver-version').text("Driver: " + driver_version);
            $('#driver-version').data("version", driver_version);
            $('#firmware-version').text("Firmware: " + firmware_version);
            $('#firmware-version').data("version", firmware_version);
        }
        $.gen_content($(this));
    });


    $.gen_content = function (elem) {
        // instructions need all options selected.
        if ($('#cann-version').val() != "na") {
            $('#install-instructions').show();
        } else {
            $('#install-instructions').hide();
            return
        }

        var options = {};
        elem.parent().parent().children().each(function () {
            var elem = $(this).find(".selected").each(function () {
                var id = $(this).attr("id").split("-");
                var category = id[0];
                var value = id[1];
                if (value == "version") {
                    if (category == "cann")
                        value = $(this).val();
                    else
                        value = $(this).data('version');
                }
                options[category] = value;
            });
        });

        // install os dependency.
        if (options['os'] == 'ubuntu') {
            $('#install-dependencies-ubuntu').show();
            $('#install-dependencies-openeuler').hide();
        } else {
            $('#install-dependencies-ubuntu').hide();
            $('#install-dependencies-openeuler').show();
        }

        var driver_url = package_info[options['cann']][options['npu']][options['arch']].driver_url;
        var firmware_url = package_info[options['cann']][options['npu']].firmware_url;
        var cann_url = package_info[options['cann']][options['arch']].url;
        var kernel_url = package_info[options['cann']][options['npu']].kernel_url;

        var parts = driver_url.split("/");
        var driver_name = parts[parts.length - 1];
        parts = firmware_url.split("/");
        var firmware_name = parts[parts.length - 1];
        parts = cann_url.split("/");
        var cann_name = parts[parts.length - 1];
        parts = kernel_url.split("/");
        var kernel_name = parts[parts.length - 1];

        // download and install driver
        $('#install_drvier').html('wget "' + driver_url + '"<br>sudo sh ' + driver_name + ' --full --install-for-all');

        // download and install firmware
        $('#install_firmware').html('wget "' + firmware_url + '"<br>sudo sh ' + firmware_name + ' --full');

        // download and install cann
        $('#install_cann').html('wget "' + cann_url + '"<br>sh ' + cann_name + ' --install');

        // download and install kernel
        $('#install_kernel').html('wget "' + kernel_url + '"<br>sh ' + kernel_name + ' --install');
    }
});
