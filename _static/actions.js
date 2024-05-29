$(document).ready(function () {
    // Version matches.
    let version_map = new Map();
    version_map.set('8.0.RC2.alpha002', ['23.0.3', '7.1.0.5.220']);
    version_map.set('8.0.RC2.alpha001', ['23.0.3', '7.1.0.5.220']);
    version_map.set('8.0.RC1.beta1', ['24.1.rc1', '7.1.0.6.220']);
    version_map.set('8.0.RC1.alpha003', ['23.0.3', '7.1.0.5.220']);
    version_map.set('8.0.RC1.alpha002', ['23.0.3', '7.1.0.5.220']);
    version_map.set('8.0.RC1.alpha001', ['23.0.3', '7.1.0.5.220']);
    version_map.set('7.0.0.beta1', ['23.0.3', '7.1.0.5.220']);
    version_map.set('7.0.0.alpha003', ['23.0.rc3', '6.4.0.4.220']);
    version_map.set('7.0.0.alpha002', ['23.0.rc3', '6.4.0.4.220']);
    version_map.set('7.0.RC1.beta1', ['23.0.rc3', '6.4.0.4.220']);

    let cann_version_select = $('#cann-version');
    version_map.forEach((value, key) => {
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
        $('#driver-version').removeData("version");
        $('#firmware-version').text("Firmware");
        $('#firmware-version').removeData("version");

        if ($(this).val() != "na") {
            $(this).addClass("selected");
            $('#driver-version').addClass("selected");
            $('#firmware-version').addClass("selected");

            let cann_version = $(this).val();
            let driver_version = version_map.get(cann_version)[0];
            let firmware_version = version_map.get(cann_version)[1];
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

        // download and install driver
        $('#install_drvier').html('wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend HDK/Ascend HDK ' + options['driver'].toUpperCase() + '/Ascend-hdk-' + options['npu'] + '-npu-driver_' + options['driver'] + '_linux-' + options['arch'].replace(/_/g, '-') + '.run"<br>sudo sh Ascend-hdk-' + options['npu'] + '-npu-driver_' + options['driver'] + '_linux-' + options['arch'].replace(/_/g, '-') + '.run --full --install-for-all');

        // download and install firmware
        $('#install_firmware').html('wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend HDK/Ascend HDK ' + options['driver'].toUpperCase() + '/Ascend-hdk-' + options['npu'] + '-npu-firmware_' + options['firmware'] + '.run"<br>sudo sh Ascend-hdk-' + options['npu'] + '-npu-firmware_' + options['firmware'] + '.run --full');

        // download and install cann
        $('#install_cann').html('wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN ' + options['cann'] + '/Ascend-cann-toolkit_' + options['cann'] + '_linux-' + options['arch'] + '.run"<br>sh Ascend-cann-toolkit_' + options['cann'] + '_linux-' + options['arch'] + '.run --install');
    }
});
