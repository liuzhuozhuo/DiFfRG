#pragma once

// DiFfRG
#include <DiFfRG/discretization/mesh/configuration_mesh.hh>

// standard library
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    /**
     * @brief Local cell widths along one axis of a rectangular mesh.
     *
     * The KT face batches carry one `"cell_width"` per point, the smallest width of the cell. Coefficients that scale
     * with the width along a given axis, such as the damping \f$\nu_a = c\,h_a^2\f$ of the coordinate maps in
     * invariant_coordinates.hh, need the width along that axis instead; on anisotropic cells the smallest one
     * under-damps the coarser direction.
     *
     * width(x) is the width of the cell containing \f$x\f$. A point on a face between two cells (the face points of
     * KT) gets the larger of the two widths, a point outside the axis (ghost cells) the width of the nearest end cell.
     */
    class AxisWidths
    {
    public:
      AxisWidths() = default;

      /**
       * @brief From the cell widths in order along the axis, starting at @p lower.
       */
      AxisWidths(const std::vector<double> &widths, const double lower) : m_widths(widths)
      {
        if (widths.empty()) throw std::invalid_argument("AxisWidths: no cells.");
        m_edges.reserve(widths.size() + 1);
        m_edges.push_back(lower);
        for (const auto h : widths) {
          if (!(h > 0.)) throw std::invalid_argument("AxisWidths: cell widths must be positive.");
          m_edges.push_back(m_edges.back() + h);
        }
      }

      /**
       * @brief The cells of axis @p axis of @p mesh, as RectangularMesh builds them (after `refine` refinements).
       */
      template <int dim> AxisWidths(const Config::ConfigurationMesh<dim> &mesh, const unsigned int axis)
      {
        const auto data = mesh.get_triangulation_data();
        const unsigned int n_sub = 1u << mesh.refine;
        std::vector<double> widths;
        for (const auto h : data.step_sizes[axis])
          widths.insert(widths.end(), n_sub, h / static_cast<double>(n_sub));
        *this = AxisWidths(widths, data.lower_left[axis]);
      }

      /// Width of the cell containing @p x; the larger one on a face, the nearest end cell outside the axis.
      double width(const double x) const
      {
        if (!(x > m_edges.front())) return m_widths.front();
        if (!(x < m_edges.back())) return m_widths.back();
        const auto it = std::upper_bound(m_edges.begin(), m_edges.end(), x);
        const std::size_t i = static_cast<std::size_t>(it - m_edges.begin()) - 1; // m_edges[i] <= x < m_edges[i + 1]
        double h = m_widths[i];
        const double tol = 1e-9 * h;
        if (i > 0 && x - m_edges[i] <= tol) h = std::max(h, m_widths[i - 1]);
        if (i + 1 < m_widths.size() && m_edges[i + 1] - x <= tol) h = std::max(h, m_widths[i + 1]);
        return h;
      }

      /// Lower end of the axis.
      double lower() const { return m_edges.front(); }
      /// Upper end of the axis.
      double upper() const { return m_edges.back(); }
      /// Width of the first cell.
      double first() const { return m_widths.front(); }
      /// Width of the last cell.
      double last() const { return m_widths.back(); }
      /// Number of cells.
      std::size_t size() const { return m_widths.size(); }

    private:
      std::vector<double> m_widths;
      std::vector<double> m_edges;
    };
  } // namespace FV
} // namespace DiFfRG
