#include "TROOT.h"
#include "TH2Poly.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TMath.h"
#include "TEllipse.h" // Добавлено для TEllipse
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <sstream>
#include <cstring>

// --- Функция для создания папки, если её нет ---
void ensure_directory_exists(const char* dir)
{
    struct stat info;
    if (stat(dir, &info) != 0)
    {
        #ifdef _WIN32
            _mkdir(dir);
        #else
            mkdir(dir, 0755);
        #endif
    }
}
// --- КОНЕЦ ---

void draw_honeycomb_hexagon(TH2Poly* h2p, Double_t x, Double_t y, Double_t size, Double_t angle_offset_rad = 0.0)
{
    const Double_t cos30 = TMath::Cos(TMath::Pi() / 6.0); // cos(30 deg)
    Double_t r = size / cos30; // radius from center to vertex

    Double_t px[6], py[6];
    for (int i = 0; i < 6; ++i)
    {
        Double_t angle = TMath::Pi() / 3 * i + TMath::Pi() / 6 + angle_offset_rad;
        px[i] = x + r * TMath::Cos(angle);
        py[i] = y + r * TMath::Sin(angle);
    }

    h2p->AddBin(6, px, py);
}

void calculate_local_pca(const std::vector<Double_t>& x_coords,
                         const std::vector<Double_t>& y_coords,
                         const std::vector<Double_t>& amps,
                         Double_t x_center, Double_t y_center,
                         Double_t neighborhood_radius,
                         Double_t& angle_rad, Double_t& major_axis, Double_t& minor_axis)
{
    std::vector<Double_t> local_x, local_y, local_weights;

    for (size_t i = 0; i < x_coords.size(); ++i)
    {
        Double_t dx = x_coords[i] - x_center;
        Double_t dy = y_coords[i] - y_center;
        Double_t dist = TMath::Sqrt(dx*dx + dy*dy);

        if (dist <= neighborhood_radius && amps[i] > 0)
        {
            local_x.push_back(x_coords[i]);
            local_y.push_back(y_coords[i]);
            local_weights.push_back(amps[i]);
        }
    }

    if (local_x.empty())
    {
        angle_rad = 0.0;
        major_axis = 1.0;
        minor_axis = 1.0;
        return;
    }

    Double_t mean_x = 0, mean_y = 0, total_weight = 0;
    for (size_t i = 0; i < local_x.size(); ++i)
    {
        mean_x += local_x[i] * local_weights[i];
        mean_y += local_y[i] * local_weights[i];
        total_weight += local_weights[i];
    }
    mean_x /= total_weight;
    mean_y /= total_weight;

    Double_t cov_xx = 0, cov_xy = 0, cov_yy = 0;
    for (size_t i = 0; i < local_x.size(); ++i)
    {
        Double_t dx = local_x[i] - mean_x;
        Double_t dy = local_y[i] - mean_y;
        Double_t w = local_weights[i];
        cov_xx += w * dx * dx;
        cov_xy += w * dx * dy;
        cov_yy += w * dy * dy;
    }
    cov_xx /= total_weight;
    cov_xy /= total_weight;
    cov_yy /= total_weight;

    if (cov_xy == 0)
    {
        angle_rad = (cov_xx > cov_yy) ? 0 : TMath::Pi()/2;
    }
    else
    {
        angle_rad = 0.5 * TMath::ATan2(2*cov_xy, cov_xx - cov_yy);
    }

    Double_t lambda1 = 0.5 * (cov_xx + cov_yy + TMath::Sqrt((cov_xx - cov_yy)*(cov_xx - cov_yy) + 4*cov_xy*cov_xy));
    Double_t lambda2 = 0.5 * (cov_xx + cov_yy - TMath::Sqrt((cov_xx - cov_yy)*(cov_xx - cov_yy) + 4*cov_xy*cov_xy));

    major_axis = TMath::Sqrt(lambda1);
    minor_axis = TMath::Sqrt(lambda2);

    if (major_axis < minor_axis)
    {
        std::swap(major_axis, minor_axis);
        angle_rad += TMath::Pi()/2;
    }

    Double_t min_size = 1.5;
    if (minor_axis < min_size) minor_axis = min_size;
    if (major_axis < min_size) major_axis = min_size;
}

// --- НОВАЯ ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ ВЕСА ---
Double_t compute_weight(const char* weight_type, Double_t dist_aniso)
{
    if (strcmp(weight_type, "linear") == 0)
    {
        // Вес линейно убывает от 1 до 0
        return dist_aniso <= 1.0 ? (1.0 - dist_aniso) : 0.0;
    }
    else if (strcmp(weight_type, "quad") == 0)
    {
        // Вес квадратично убывает от 1 до 0
        Double_t val = 1.0 - dist_aniso * dist_aniso;
        return dist_aniso <= 1.0 ? val : 0.0;
    }
    else if (strcmp(weight_type, "const") == 0)
    {
        // Постоянный вес внутри эллипса
        return dist_aniso <= 1.0 ? 1.0 : 0.0;
    }
    else
    { // "gauss" или любой другой вариант по умолчанию
        // Гауссовский вес
        return TMath::Exp(-1.5 * dist_aniso * dist_aniso);
    }
}
// --- КОНЕЦ ---

void get_hexagon_anisotropic_weights(const std::vector<Double_t>& x_coords,
                                     const std::vector<Double_t>& y_coords,
                                     const std::vector<Double_t>& amps,
                                     TH2F* h2f, Double_t x, Double_t y, Double_t size,
                                     Double_t angle_rad, Double_t major_axis, Double_t minor_axis,
                                     const char* weight_type,
                                     std::vector<std::pair<Int_t, Int_t>>& bins_to_fill,
                                     std::vector<Double_t>& weights_to_fill)
{
    const Double_t cos30 = TMath::Cos(TMath::Pi() / 6.0);
    Double_t r = 1.75 * size / cos30;

    Int_t n_bins_x = h2f->GetNbinsX();
    Int_t n_bins_y = h2f->GetNbinsY();

    for (Int_t ix = 1; ix <= n_bins_x; ++ix)
    {
        Double_t px = h2f->GetXaxis()->GetBinCenter(ix);
        Double_t dx = px - x;
        for (Int_t iy = 1; iy <= n_bins_y; ++iy)
        {
            Double_t py = h2f->GetYaxis()->GetBinCenter(iy);
            Double_t dy = py - y;

            Double_t dx_rot = dx * TMath::Cos(-angle_rad) - dy * TMath::Sin(-angle_rad);
            Double_t dy_rot = dx * TMath::Sin(-angle_rad) + dy * TMath::Cos(-angle_rad);

            Double_t dist_aniso = TMath::Sqrt( (dx_rot*dx_rot)/(major_axis*major_axis) + (dy_rot*dy_rot)/(minor_axis*minor_axis) );

            if (dist_aniso <= 1.0)
            {
                Double_t weight = compute_weight(weight_type, dist_aniso);

                bins_to_fill.push_back(std::make_pair(ix, iy));
                weights_to_fill.push_back(weight);
            }
        }
    }

    if (bins_to_fill.empty())
    {
        Int_t ix_closest = h2f->GetXaxis()->FindBin(x);
        Int_t iy_closest = h2f->GetYaxis()->FindBin(y);

        if (ix_closest < 1) ix_closest = 1;
        if (ix_closest > n_bins_x) ix_closest = n_bins_x;
        if (iy_closest < 1) iy_closest = 1;
        if (iy_closest > n_bins_y) iy_closest = n_bins_y;

        bins_to_fill.push_back(std::make_pair(ix_closest, iy_closest));
        weights_to_fill.push_back(1.0);
    }
}

void rotate_point(Double_t x, Double_t y, Double_t angle_rad, Double_t& xr, Double_t& yr)
{
    xr = x * TMath::Cos(angle_rad) - y * TMath::Sin(angle_rad);
    yr = x * TMath::Sin(angle_rad) + y * TMath::Cos(angle_rad);
}

void draw_ellipse(Double_t x, Double_t y, Double_t a, Double_t b, Double_t angle_rad, Int_t color = kRed, Int_t style = 2)
{
    TEllipse *ellipse = new TEllipse(x, y, a, b);
    ellipse->SetFillStyle(0); // Прозрачный
    ellipse->SetLineColor(color);
    ellipse->SetLineStyle(style);
    ellipse->SetLineWidth(1);
    ellipse->SetTheta(angle_rad * 180.0 / TMath::Pi()); // ROOT использует градусы
    ellipse->Draw();
}

void process_event(Int_t event_id, const std::vector<Double_t>& x_coords,
                   const std::vector<Double_t>& y_coords, const std::vector<Double_t>& amps)
{
    // --- 🔧 НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ---
    const Double_t pixel_size = 1.5;
    const Double_t square_bin_size = 0.5;
    const Double_t neighborhood_radius = 7.5;
    const Double_t spread_factor = 1.5;
    const Double_t min_axis_size = 1.5;
    const bool draw_ellipses = false;  // <-- ПЕРЕМЕННАЯ: true = рисовать, false = не рисовать
    const char* weight_type = "gauss"; // <-- Тип веса: "gauss", "linear", "quad", "const"
    // --- КОНЕЦ НАСТРОЕК ---

    const Double_t cos30 = TMath::Cos(TMath::Pi() / 6.0);
    const Double_t hex_radius = pixel_size / cos30;

    const Double_t x_min = -40.0;
    const Double_t x_max = 40.0;
    const Double_t y_min = -40.0;
    const Double_t y_max = 40.0;

    const Double_t hex_rotation_rad = 30.0 * TMath::Pi() / 180.0;

    TH2Poly *h_hex = new TH2Poly("h_hex", "Hexagonal Bins;X [cm];Y [cm]", x_min, x_max, y_min, y_max);
    Double_t total_amp_original = 0.0;

    for (size_t i = 0; i < x_coords.size(); ++i)
    {
        if (x_coords[i] >= x_min && x_coords[i] <= x_max && y_coords[i] >= y_min && y_coords[i] <= y_max)
        {
            draw_honeycomb_hexagon(h_hex, x_coords[i], y_coords[i], pixel_size, hex_rotation_rad);
            h_hex->Fill(x_coords[i], y_coords[i], amps[i]);
            total_amp_original += amps[i];
        }
    }

    TH2F *h_sq = new TH2F("h_sq", "Square Bins;X [cm];Y [cm]",
                          (x_max - x_min) / square_bin_size, x_min, x_max,
                          (y_max - y_min) / square_bin_size, y_min, y_max);

    Int_t n_bins_x = h_sq->GetNbinsX();
    Int_t n_bins_y = h_sq->GetNbinsY();
    std::vector<std::vector<Double_t>> temp_content(n_bins_x + 1, std::vector<Double_t>(n_bins_y + 1, 0.0));

    for (size_t i = 0; i < x_coords.size(); ++i)
    {
        if (x_coords[i] >= x_min && x_coords[i] <= x_max && y_coords[i] >= y_min && y_coords[i] <= y_max)
        {
            Double_t angle_rad, major_axis, minor_axis;
            calculate_local_pca(x_coords, y_coords, amps,
                                x_coords[i], y_coords[i],
                                neighborhood_radius,
                                angle_rad, major_axis, minor_axis);

            std::vector<std::pair<Int_t, Int_t>> bins_to_fill;
            std::vector<Double_t> weights_to_fill;

            // --- ПЕРЕДАЁМ weight_type в функцию ---
            get_hexagon_anisotropic_weights(x_coords, y_coords, amps, h_sq, x_coords[i], y_coords[i], pixel_size,
                                            angle_rad, major_axis, minor_axis, weight_type, bins_to_fill, weights_to_fill);

            Double_t sum_weights = 0.0;
            for (auto w : weights_to_fill) sum_weights += w;
            if (sum_weights == 0.0) continue;

            for (size_t k = 0; k < bins_to_fill.size(); ++k)
            {
                Int_t ix = bins_to_fill[k].first;
                Int_t iy = bins_to_fill[k].second;
                Double_t weight = weights_to_fill[k];
                Double_t contribution = amps[i] * (weight / sum_weights);
                temp_content[ix][iy] += contribution;
            }
        }
    }

    for (Int_t ix = 1; ix <= n_bins_x; ++ix)
    {
        for (Int_t iy = 1; iy <= n_bins_y; ++iy)
        {
            h_sq->SetBinContent(ix, iy, temp_content[ix][iy]);
        }
    }

    Double_t total_amp_spread = h_sq->Integral();

    // Rotated histograms
    Double_t angle_rad = 7.5 * TMath::Pi() / 180.0;

    // --- Обновлённые заголовки с типом веса и радиусом ---
    TH2Poly *h_hex_rot = new TH2Poly("h_hex_rot", Form("Rotated Hexagonal Bins (%s, r=%.1f);X [cm];Y [cm]", weight_type, neighborhood_radius), x_min, x_max, y_min, y_max);
    std::vector<Double_t> x_rot, y_rot;
    for (size_t i = 0; i < x_coords.size(); ++i)
    {
        Double_t xr, yr;
        rotate_point(x_coords[i], y_coords[i], angle_rad, xr, yr);
        x_rot.push_back(xr);
        y_rot.push_back(yr);
        if (xr >= x_min && xr <= x_max && yr >= y_min && yr <= y_max)
        {
            draw_honeycomb_hexagon(h_hex_rot, xr, yr, pixel_size, hex_rotation_rad);
            h_hex_rot->Fill(xr, yr, amps[i]);
        }
    }

    TH2F *h_sq_rot = new TH2F("h_sq_rot", Form("Rotated Square Bins (%s, r=%.1f);X [cm];Y [cm]", weight_type, neighborhood_radius),
                              (x_max - x_min) / square_bin_size, x_min, x_max,
                              (y_max - y_min) / square_bin_size, y_min, y_max);

    // --- Анизотропная перепикселизация для повернутой сетки ---
    std::vector<std::vector<Double_t>> temp_content_rot(n_bins_x + 1, std::vector<Double_t>(n_bins_y + 1, 0.0));

    // --- Сохраняем локальные параметры для визуализации ---
    std::vector<Double_t> angles_local;
    std::vector<Double_t> major_axes_local;
    std::vector<Double_t> minor_axes_local;
    // --- КОНЕЦ ---

    for (size_t i = 0; i < x_rot.size(); ++i)
    {
        if (x_rot[i] >= x_min && x_rot[i] <= x_max && y_rot[i] >= y_min && y_rot[i] <= y_max)
        {
            Double_t angle_rad_local, major_axis, minor_axis;
            calculate_local_pca(x_rot, y_rot, amps,
                                x_rot[i], y_rot[i],
                                neighborhood_radius,
                                angle_rad_local, major_axis, minor_axis);

            // --- Сохраняем параметры ---
            angles_local.push_back(angle_rad_local);
            major_axes_local.push_back(major_axis);
            minor_axes_local.push_back(minor_axis);
            // --- КОНЕЦ ---

            std::vector<std::pair<Int_t, Int_t>> bins_to_fill;
            std::vector<Double_t> weights_to_fill;

            // --- ПЕРЕДАЁМ weight_type в функцию ---
            get_hexagon_anisotropic_weights(x_rot, y_rot, amps, h_sq_rot, x_rot[i], y_rot[i], pixel_size,
                                            angle_rad_local, major_axis, minor_axis, weight_type, bins_to_fill, weights_to_fill);

            Double_t sum_weights = 0.0;
            for (auto w : weights_to_fill) sum_weights += w;
            if (sum_weights == 0.0) continue;

            for (size_t k = 0; k < bins_to_fill.size(); ++k)
            {
                Int_t ix = bins_to_fill[k].first;
                Int_t iy = bins_to_fill[k].second;
                Double_t weight = weights_to_fill[k];
                Double_t contribution = amps[i] * (weight / sum_weights);
                temp_content_rot[ix][iy] += contribution;
            }
        }
    }

    for (Int_t ix = 1; ix <= n_bins_x; ++ix)
    {
        for (Int_t iy = 1; iy <= n_bins_y; ++iy)
        {
            h_sq_rot->SetBinContent(ix, iy, temp_content_rot[ix][iy]);
        }
    }

    Double_t total_amp_spread_rot = h_sq_rot->Integral();

    // Canvas for rotated histograms
    // --- Обновляем название холста ---
    TCanvas *c2 = new TCanvas("c2", Form("Event %d - Rotated Histograms (%s, r=%.1f)", event_id, weight_type, neighborhood_radius), 1200, 600);
    c2->Divide(2, 1);

    c2->cd(1);
    h_hex_rot->Draw("COLZ");
    gPad->SetRightMargin(0.15);

    c2->cd(2);
    h_sq_rot->Draw("COLZ");
    gPad->SetRightMargin(0.15);

    // --- НАЧАЛО: Рисуем эллипсы (только если draw_ellipses = true) ---
    if (draw_ellipses)
    {
        for (size_t i = 0; i < x_rot.size(); ++i)
        {
            if (x_rot[i] >= x_min && x_rot[i] <= x_max && y_rot[i] >= y_min && y_rot[i] <= y_max)
            {
                draw_ellipse(x_rot[i], y_rot[i], major_axes_local[i], minor_axes_local[i], angles_local[i]);
            }
        }
    }
    // --- КОНЕЦ ---

    // Clean canvas
    TCanvas *c3 = new TCanvas("c3", Form("Event %d - Rotated Square Histogram (Clean, %s, r=%.1f)", event_id, weight_type, neighborhood_radius), 600, 600);
    c3->SetBorderSize(0);
    c3->SetFrameBorderSize(0);
    h_sq_rot->Draw("COLZ");
    // --- Рисуем эллипсы на чистом холсте тоже (только если draw_ellipses = true) ---
    if (draw_ellipses)
    {
        for (size_t i = 0; i < x_rot.size(); ++i)
        {
            if (x_rot[i] >= x_min && x_rot[i] <= x_max && y_rot[i] >= y_min && y_rot[i] <= y_max)
            {
                draw_ellipse(x_rot[i], y_rot[i], major_axes_local[i], minor_axes_local[i], angles_local[i]);
            }
        }
    }
    // --- КОНЕЦ ---
    c3->Update();

    // --- СОХРАНЕНИЕ В ПАПКУ С ИМЕНАМИ ---
    std::string prefix = Form("event_%d_%s_%.0fr", event_id, weight_type, neighborhood_radius);
    std::string path1 = "fullfile/" + prefix + "_rotated_histograms.png";
    std::string path2 = "fullfile/" + prefix + "_rotated_square_clean.png";

    c2->SaveAs(path1.c_str());
    c3->SaveAs(path2.c_str());
    // --- КОНЕЦ ---

    // Save rotated square grid data
    std::string output_filename = "fullfile/square_grid_event_" + std::to_string(event_id) + ".txt";
    std::ofstream output_file(output_filename);
    if (output_file.is_open())
    {
        output_file << "X Y Amplitude" << std::endl;
        for (Int_t ix = 1; ix <= n_bins_x; ++ix)
        {
            Double_t x_center = h_sq_rot->GetXaxis()->GetBinCenter(ix);
            for (Int_t iy = 1; iy <= n_bins_y; ++iy)
            {
                Double_t y_center = h_sq_rot->GetYaxis()->GetBinCenter(iy);
                Double_t content = h_sq_rot->GetBinContent(ix, iy);
                if (content != 0)
                {
                    output_file << x_center << " " << y_center << " " << content << std::endl;
                }
            }
        }
        output_file.close();
        std::cout << "Event " << event_id << ": Rotated square grid data saved to '" << output_filename << "'" << std::endl;
    }

    // Check integrals
    std::cout << "\n--- Event " << event_id << " - Amplitude Integral Check ---" << std::endl;
    std::cout << "Total original amplitude (before spreading): " << total_amp_original << std::endl;
    std::cout << "Total amplitude in square histogram (after spreading): " << total_amp_spread << std::endl;
    std::cout << "Total amplitude in rotated square histogram: " << total_amp_spread_rot << std::endl;

    const Double_t tolerance = 1e-6;
    if (std::abs(total_amp_original - total_amp_spread) < tolerance)
    {
        std::cout << "Check passed: Original and spread integrals match within tolerance." << std::endl;
    }
    else
    {
        std::cout << "Check failed: Original and spread integrals do not match!" << std::endl;
        std::cout << "Difference: " << std::abs(total_amp_original - total_amp_spread) << std::endl;
    }

    if (std::abs(total_amp_original - total_amp_spread_rot) < tolerance)
    {
        std::cout << "Check passed: Original and rotated spread integrals match within tolerance." << std::endl;
    }
    else
    {
        std::cout << "Check failed: Original and rotated spread integrals do not match!" << std::endl;
        std::cout << "Difference: " << std::abs(total_amp_original - total_amp_spread_rot) << std::endl;
    }

    // --- Обновляем сообщение ---
    std::cout << "Images saved as '" << path1 << "' and '" << path2 << "'" << std::endl;

    // Очищаем память
    delete h_hex;
    delete h_sq;
    delete h_hex_rot;
    delete h_sq_rot;
    delete c2;
    delete c3;
}

void multifile_honeygrid()
{
    // --- Создаём папку fullfile ---
    ensure_directory_exists("fullfile");
    // --- КОНЕЦ ---

    const char* filename = "020321.cleanout_14_7.0fix_001.txt";
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Cannot open file " << filename << std::endl;
        return;
    }

    std::string line;
    Int_t current_event_id = -1;
    std::vector<Double_t> current_x_coords, current_y_coords, current_amps;

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string first_token;
        iss >> first_token;

        // Проверяем, является ли строка заголовком события
        // Предполагаем, что заголовок содержит 4 поля: ID, ID, строка, число
        std::istringstream check_line_stream(line);
        std::vector<std::string> tokens;
        std::string token;
        while (check_line_stream >> token)
        {
            tokens.push_back(token);
        }

        // Если строка содержит 4 токена и третий токен содержит ':', это заголовок события
        if (tokens.size() == 4 && tokens[2].find(':') != std::string::npos)
        {
            // Обработать предыдущее событие, если оно есть
            if (current_event_id != -1)
            {
                std::cout << "Processing event " << current_event_id << " with " << current_x_coords.size() << " hits..." << std::endl;
                process_event(current_event_id, current_x_coords, current_y_coords, current_amps);

                // Очистить векторы для следующего события
                current_x_coords.clear();
                current_y_coords.clear();
                current_amps.clear();
            }

            // Установить новый ID события
            current_event_id = std::stoi(tokens[0]);
        }
        else if (tokens.size() >= 5)
        {
            // Это строка данных: cluster pixel x y amp
            try
            {
                Int_t cluster, pixel;
                Double_t x, y, amp;
                std::istringstream data_stream(line);
                data_stream >> cluster >> pixel >> x >> y >> amp;

                current_x_coords.push_back(x);
                current_y_coords.push_back(y);
                current_amps.push_back(amp);
            }
            catch (...)
            {
                // Пропустить некорректную строку
                continue;
            }
        }
        // Пропускаем строки, которые не подходят ни под заголовок, ни под данные
    }

    // Обработать последнее событие
    if (current_event_id != -1)
    {
        std::cout << "Processing final event " << current_event_id << " with " << current_x_coords.size() << " hits..." << std::endl;
        process_event(current_event_id, current_x_coords, current_y_coords, current_amps);
    }

    file.close();

    if (current_event_id == -1)
    {
        std::cerr << "No events found in file." << std::endl;
    }
    else
    {
        std::cout << "All events processed successfully." << std::endl;
    }
}